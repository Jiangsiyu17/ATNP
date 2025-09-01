# -*- coding: utf-8 -*-
# build_index_fixed.py

import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import pickle
import numpy as np
import hnswlib
import gensim
from tqdm import tqdm

# 设置 Django 环境
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ATNP.settings")
import django
django.setup()

from db_utils import load_plant_spectra_from_folder
from matchms.filtering import normalize_intensities
from spec2vec import SpectrumDocument
from spec2vec.vector_operations import calc_vector

# 路径配置
PLANT_POS_DIR = "/data2/jiangsiyu/ATNP_Database/Herbs/pos"
PLANT_NEG_DIR = "/data2/jiangsiyu/ATNP_Database/Herbs/neg"
MODEL_POS = "/data2/jiangsiyu/ATNP_Database/model/Ms2Vec_allGNPSpositive.hdf5"
MODEL_NEG = "/data2/jiangsiyu/ATNP_Database/model/Ms2Vec_allGNPSnegative.hdf5"
OUT_PICKLE_POS = "/data2/jiangsiyu/ATNP_Database/model/herbs_spectra_pos.pickle"
OUT_PICKLE_NEG = "/data2/jiangsiyu/ATNP_Database/model/herbs_spectra_neg.pickle"
OUT_INDEX_POS = "/data2/jiangsiyu/ATNP_Database/model/herbs_index_pos.bin"
OUT_INDEX_NEG = "/data2/jiangsiyu/ATNP_Database/model/herbs_index_neg.bin"

# 多进程加载模型
model = None

def init_worker(model_path):
    """每个进程初始化时加载模型"""
    global model
    print(f"[PID {os.getpid()}] Loading model ...")
    model = gensim.models.Word2Vec.load(model_path)


def calc_ms2vec_vector_mp(spec):
    """计算单个谱图的向量，返回 (vec, spec, reason)"""
    global model
    # 归一化强度
    spec_norm = normalize_intensities(spec)

    # 补全 metadata
    meta = spec_norm.metadata.copy()  # 确保独立 dict
    meta["latin_name"] = meta.get("latin_name", "unknown")
    meta["chinese_name"] = meta.get("chinese_name", "unknown")
    meta["tissue"] = meta.get("tissue", "unknown")
    spec_norm.metadata = meta

    if spec_norm.intensities is None or len(spec_norm.intensities) == 0:
        return None, spec_norm, "no peaks"

    doc = SpectrumDocument(spec_norm, n_decimals=2)
    vec = calc_vector(model, doc, allowed_missing_percentage=100)

    if vec is None:
        return None, spec_norm, "calc_vector failed"
    return vec, spec_norm, None


def process_plant_spectra(mgf_dir: str, model_path: str, pickle_path: str, index_path: str, max_workers: int = None):
    """加载谱图 -> 计算向量 -> 过滤 -> 保存pickle和索引"""
    print(f"\n📂 Loading spectra from {mgf_dir} ...")
    ionmode = "positive" if "pos" in mgf_dir.lower() else "negative"
    all_spectra = load_plant_spectra_from_folder(mgf_dir, ionmode=ionmode)
    print(f"✔️ Loaded {len(all_spectra)} spectra with metadata.")

    raw_vectors = []
    raw_spectra = []
    filter_logs = []

    print(f"⚙️  Start vectorization with max_workers={max_workers} ...")
    with ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker, initargs=(model_path,)) as executor:
        results_iter = executor.map(calc_ms2vec_vector_mp, all_spectra)

        for i, (vec, spec, reason) in enumerate(tqdm(results_iter, total=len(all_spectra), desc="Vectorizing")):
            title = spec.metadata.get("title", f"spectrum_{i}")
            source_file = spec.metadata.get("source_file", "unknown_file")
            if vec is not None:
                raw_vectors.append(vec)
                raw_spectra.append(spec)
            else:
                filter_logs.append(f"{source_file}\t{title}\t{reason}")

    print(f"➡️  Got {len(raw_spectra)} non-empty spectra with vectors.")

    # 转换为数组并过滤全零向量
    vectors = np.array(raw_vectors).astype("float32")
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    valid_mask = (norm > 0).ravel()

    vectors = vectors[valid_mask]
    valid_spectra = [spec for i, spec in enumerate(raw_spectra) if valid_mask[i]]

    # 保存日志
    log_path = pickle_path.replace(".pickle", "_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("source_file\ttitle\treason\n")
        for line in filter_logs:
            f.write(line + "\n")
    print(f"📝 Saved filter log to {log_path}")

    # 💾 保存谱图对象
    with open(pickle_path, "wb") as f:
        pickle.dump(valid_spectra, f)
    print(f"✅ Saved spectra to {pickle_path}")

    # 构建并保存 HNSW 索引
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
