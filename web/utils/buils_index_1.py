# -*- coding: utf-8 -*-
# build_index.py 多进程版本，加载植物谱图并构建向量索引（PEPMASS=None 自动修复）

import os
import sys
from concurrent.futures import ProcessPoolExecutor
import pickle
import numpy as np
import hnswlib
import gensim
from tqdm import tqdm
from pathlib import Path
import tempfile

# ---------------- 设置 Django 环境 ----------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ATNP.settings")
import django
django.setup()

from matchms import Spectrum
from matchms.importing import load_from_mgf

# ---------------- 路径配置 ----------------
PLANT_POS_DIR = "/data2/jiangsiyu/ATNP_Database/Herbs/cleaned_mgf/pos"
PLANT_NEG_DIR = "/data2/jiangsiyu/ATNP_Database/Herbs/cleaned_mgf/neg"
MODEL_POS = "/data2/jiangsiyu/ATNP_Database/model/Ms2Vec_allGNPSpositive.hdf5"
MODEL_NEG = "/data2/jiangsiyu/ATNP_Database/model/Ms2Vec_allGNPSnegative.hdf5"
OUT_PICKLE_POS = "/data2/jiangsiyu/ATNP_Database/model/herbs_spectra_pos_2.pickle"
OUT_PICKLE_NEG = "/data2/jiangsiyu/ATNP_Database/model/herbs_spectra_neg_2.pickle"
OUT_INDEX_POS = "/data2/jiangsiyu/ATNP_Database/model/herbs_index_pos_2.bin"
OUT_INDEX_NEG = "/data2/jiangsiyu/ATNP_Database/model/herbs_index_neg_2.bin"

# ---------------- 多进程全局模型 ----------------
model = None

def init_worker(model_path):
    """每个进程初始化时加载模型"""
    global model
    print(f"[PID {os.getpid()}] Loading model ...")
    model = gensim.models.Word2Vec.load(model_path)

def calc_ms2vec_vector_mp(spec):
    """计算单个谱图的向量，返回 (vec, reason)"""
    from matchms.filtering import normalize_intensities
    from spec2vec import SpectrumDocument
    from spec2vec.vector_operations import calc_vector

    global model
    spec_norm = normalize_intensities(spec)

    # 没有峰，直接过滤
    if spec_norm.intensities is None or len(spec_norm.intensities) == 0:
        return None, "no peaks"

    doc = SpectrumDocument(spec_norm, n_decimals=2)
    vec = calc_vector(model, doc, allowed_missing_percentage=100)

    if vec is None:
        return None, "calc_vector failed"
    return vec, None

# ---------------- 安全加载 MGF ----------------
def safe_load_from_mgf(file_path):
    """安全加载 MGF：先替换 PEPMASS=None，再交给 load_from_mgf"""
    # 读取原文件
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 替换 PEPMASS=None
    new_lines = []
    for line in lines:
        if line.strip().upper().startswith("PEPMASS=") and "NONE" in line.upper():
            new_lines.append("PEPMASS=0\n")
        else:
            new_lines.append(line)

    # 写入临时文件
    with tempfile.NamedTemporaryFile("w+", delete=False, encoding="utf-8", suffix=".mgf") as tmp:
        tmp.writelines(new_lines)
        tmp_path = tmp.name

    # 加载谱图
    spectra = list(load_from_mgf(tmp_path))

    # 删除临时文件
    os.unlink(tmp_path)

    return spectra

def load_all_spectra_from_folder(mgf_dir):
    """遍历文件夹，安全加载所有 MGF 谱图"""
    all_spectra = []
    for mgf_file in Path(mgf_dir).glob("*.mgf"):
        all_spectra.extend(safe_load_from_mgf(mgf_file))
    return all_spectra

# ---------------- 主处理函数 ----------------
def process_plant_spectra(mgf_dir: str, model_path: str, pickle_path: str, index_path: str, max_workers: int = None):
    """加载谱图 -> 去重 -> 计算向量 -> 过滤 -> 保存 pickle 和索引"""
    print(f"\n📂 Loading spectra from {mgf_dir} ...")
    all_spectra = load_all_spectra_from_folder(mgf_dir)
    print(f"✔️ Loaded {len(all_spectra)} spectra (PEPMASS fixed if needed).")

    # --- 去重，根据 (latin_name, precursor_mz) 保留强度最高 ---
    spec_dict = {}
    for spec in all_spectra:
        meta = spec.metadata
        key = (meta.get("latin_name"), round(meta.get("precursor_mz", 0), 4))
        total_intensity = spec.intensities.sum() if spec.intensities is not None else 0
        if key not in spec_dict or total_intensity > spec_dict[key].intensities.sum():
            spec_dict[key] = spec
    all_spectra = list(spec_dict.values())
    print(f"🔹 After precursor_mz filtering: {len(all_spectra)} spectra remain")

    raw_vectors = []
    raw_spectra = []
    filter_logs = []

    print(f"⚙️  Start vectorization with max_workers={max_workers} ...")
    with ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker, initargs=(model_path,)) as executor:
        results_iter = executor.map(calc_ms2vec_vector_mp, all_spectra)
        for i, (vec, reason) in enumerate(tqdm(results_iter, total=len(all_spectra), desc="Vectorizing")):
            spec = all_spectra[i]
            title = spec.metadata.get("title", f"spectrum_{i}")
            source_file = spec.metadata.get("source_file", "unknown_file")
            if vec is not None:
                raw_vectors.append(vec)
                raw_spectra.append(spec)
            else:
                filter_logs.append(f"{source_file}\t{title}\t{reason}")

    print(f"➡️ Got {len(raw_spectra)} non-empty spectra with vectors.")

    # 转换为数组并过滤全零向量
    vectors = np.array(raw_vectors).astype("float32")
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    valid_mask = (norm > 0).ravel()

    vectors = vectors[valid_mask]
    valid_spectra = [spec for i, spec in enumerate(raw_spectra) if valid_mask[i]]

    # 被过滤掉的零向量也记日志
    for i, keep in enumerate(valid_mask):
        if not keep:
            title = raw_spectra[i].metadata.get("title", f"spectrum_{i}")
            source_file = raw_spectra[i].metadata.get("source_file", "unknown_file")
            filter_logs.append(f"{source_file}\t{title}\tzero vector")

    print(f"✅ Final spectra saved to pickle: {len(valid_spectra)} / {len(all_spectra)}")

    # 💾 保存谱图对象（保证与索引一一对应）
    with open(pickle_path, "wb") as f:
        pickle.dump(valid_spectra, f)

    # 💾 保存日志文件
    log_path = pickle_path.replace(".pickle", "_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("source_file\ttitle\treason\n")
        for line in filter_logs:
            f.write(line + "\n")
    print(f"📝 Saved filter log to {log_path}")

    # 构建并保存索引
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    dim = vectors.shape[1]
    index = hnswlib.Index(space="l2", dim=dim)
    index.init_index(max_elements=len(vectors), ef_construction=400, M=64)
    index.add_items(vectors, np.arange(len(vectors)))
    index.set_ef(300)
    index.save_index(index_path)
    print(f"✅ Saved index to {index_path}")


if __name__ == "__main__":
    process_plant_spectra(PLANT_POS_DIR, MODEL_POS, OUT_PICKLE_POS, OUT_INDEX_POS, max_workers=None)
    process_plant_spectra(PLANT_NEG_DIR, MODEL_NEG, OUT_PICKLE_NEG, OUT_INDEX_NEG, max_workers=None)