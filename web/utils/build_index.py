# -*- coding: utf-8 -*-
# build_index.py 多进程版本，加载植物谱图并构建向量索引（保证pkl和bin严格对应，并记录过滤日志）

import os
from pathlib import Path
import sys
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


# 路径配置
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLANT_POS_DIR = str(PROJECT_ROOT / "Herbs" / "pos")
PLANT_NEG_DIR = str(PROJECT_ROOT / "Herbs" / "neg")
MODEL_POS = str(PROJECT_ROOT / "model" / "Ms2Vec_allGNPSpositive.hdf5")
MODEL_NEG = str(PROJECT_ROOT / "model" / "Ms2Vec_allGNPSnegative.hdf5")
OUT_PICKLE_POS = str(PROJECT_ROOT / "model" / "herbs_spectra_pos.pickle")
OUT_PICKLE_NEG = str(PROJECT_ROOT / "model" / "herbs_spectra_neg.pickle")
OUT_INDEX_POS = str(PROJECT_ROOT / "model" / "herbs_index_pos.bin")
OUT_INDEX_NEG = str(PROJECT_ROOT / "model" / "herbs_index_neg.bin")


# 多进程加载模型
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


def process_plant_spectra(mgf_dir: str, model_path: str, pickle_path: str, index_path: str, max_workers: int = None):
    """加载谱图 -> 计算向量 -> 过滤 -> 保存 pickle(谱图+向量) 和索引"""
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

        for i, (vec, reason) in enumerate(tqdm(results_iter, total=len(all_spectra), desc="Vectorizing")):
            spec = all_spectra[i]
            title = spec.metadata.get("title", f"spectrum_{i}")
            source_file = spec.metadata.get("source_file", "unknown_file")
            spec.metadata["herb_id"] = f"{ionmode}_{i:06d}"
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

    # 把被过滤掉的零向量也记日志
    for i, keep in enumerate(valid_mask):
        if not keep:
            title = raw_spectra[i].metadata.get("title", f"spectrum_{i}")
            source_file = raw_spectra[i].metadata.get("source_file", "unknown_file")
            filter_logs.append(f"{source_file}\t{title}\tzero vector")

    print(f"✅ Valid non-zero vectors: {len(valid_spectra)} / {len(all_spectra)}")

    # 💾 保存谱图对象 + 向量
    print(f"💾 Saving spectra+vectors to {pickle_path}")
    data_to_save = [
        {"spectrum": spec, "vector": vec}
        for spec, vec in zip(valid_spectra, vectors)
    ]
    with open(pickle_path, "wb") as f:
        pickle.dump(data_to_save, f)

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
