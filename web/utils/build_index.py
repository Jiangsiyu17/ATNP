# -*- coding: utf-8 -*-
# build_index.py 多进程版本，加载植物谱图并构建向量索引

import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import pickle
import numpy as np
import hnswlib
import gensim
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ATNP.settings")
import django
django.setup()

from db_utils import load_plant_spectra_from_folder


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
    global model
    print(f"[PID {os.getpid()}] Loading model ...")
    model = gensim.models.Word2Vec.load(model_path)


def calc_ms2vec_vector_mp(spec):
    from matchms.filtering import normalize_intensities
    from spec2vec import SpectrumDocument
    from spec2vec.vector_operations import calc_vector

    global model
    spec_norm = normalize_intensities(spec)
    if spec_norm.intensities is None or len(spec_norm.intensities) == 0:
        return None
    doc = SpectrumDocument(spec_norm, n_decimals=2)
    vec = calc_vector(model, doc, allowed_missing_percentage=100)
    return vec


def process_plant_spectra(mgf_dir: str, model_path: str, pickle_path: str, index_path: str, max_workers: int = None):
    print(f"\n📂 Loading spectra from {mgf_dir} ...")
    ionmode = "positive" if "pos" in mgf_dir.lower() else "negative"
    all_spectra = load_plant_spectra_from_folder(mgf_dir, ionmode=ionmode)
    print(f"✔️ Loaded {len(all_spectra)} spectra with metadata.")

    vectors = []
    valid_spectra = []

    print(f"⚙️  Start vectorization with max_workers={max_workers} ...")
    with ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker, initargs=(model_path,)) as executor:
        results_iter = executor.map(calc_ms2vec_vector_mp, all_spectra)

        for i, vec in enumerate(tqdm(results_iter, total=len(all_spectra), desc="Vectorizing")):
            if vec is not None:
                vectors.append(vec)
                valid_spectra.append(all_spectra[i])

    print(f"✅ Valid vectors: {len(valid_spectra)} / {len(all_spectra)}")
    
    # 保存谱图对象
    print(f"💾 Saving spectra to {pickle_path}")
    with open(pickle_path, "wb") as f:
        pickle.dump(valid_spectra, f)

    # 构建向量索引
    xb = np.array(vectors).astype("float32")
    norm = np.linalg.norm(xb, axis=1, keepdims=True)
    valid = (norm > 0).ravel()
    if not np.all(valid):
        print(f"⚠️ Detected {np.sum(~valid)} invalid vectors (zero norm), skipping them.")
    xb = xb[valid]
    xb /= np.linalg.norm(xb, axis=1, keepdims=True)

    dim = xb.shape[1]
    index = hnswlib.Index(space="l2", dim=dim)
    index.init_index(max_elements=len(xb), ef_construction=400, M=64)
    index.add_items(xb, np.arange(len(xb)))
    index.set_ef(300)
    index.save_index(index_path)
    print(f"✅ Saved index to {index_path}")


if __name__ == "__main__":
    process_plant_spectra(PLANT_POS_DIR, MODEL_POS, OUT_PICKLE_POS, OUT_INDEX_POS, max_workers=None)
    process_plant_spectra(PLANT_NEG_DIR, MODEL_NEG, OUT_PICKLE_NEG, OUT_INDEX_NEG, max_workers=None)
