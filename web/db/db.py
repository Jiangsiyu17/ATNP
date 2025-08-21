# -*- coding: utf-8 -*-
import numpy as np
import pickle
import hnswlib
import gensim
from tqdm import tqdm
from spec2vec import SpectrumDocument
from spec2vec.vector_operations import calc_vector
from matchms.filtering import normalize_intensities
import os


def build_index(pickle_in, pickle_out, bin_out, model_path, top_n=1000):
    """
    从原始pkl中取前top_n条生成新的pkl，并根据这些数据生成bin文件
    自动对谱图进行normalize_intensities，避免峰强度不归一化报错
    """
    # 加载 Word2Vec 模型
    model = gensim.models.Word2Vec.load(model_path)
    calc_ms2vec_vector = lambda x: calc_vector(
        model, SpectrumDocument(x, n_decimals=2), allowed_missing_percentage=100
    )

    # 读取原始 pickle
    with open(pickle_in, "rb") as f:
        reference = pickle.load(f)

    # 只取前 top_n 条
    reference = reference[:top_n]

    # 保存新的pkl
    with open(pickle_out, "wb") as f:
        pickle.dump(reference, f)
    print(f"✅ Saved new pickle: {pickle_out} ({len(reference)} spectra)")

    # 计算向量
    reference_vector = []
    for s in tqdm(reference, desc=f"Vectorizing {os.path.basename(pickle_out)}"):
        s = normalize_intensities(s)   # <--- 自动归一化
        reference_vector.append(calc_ms2vec_vector(s))

    xb = np.array(reference_vector).astype("float32")
    xb_len = np.linalg.norm(xb, axis=1, keepdims=True)
    xb = xb / xb_len

    dim = xb.shape[1]  # 300
    num_elements = len(xb)
    ids = np.arange(num_elements)

    # 构建 HNSW 索引
    p = hnswlib.Index(space="l2", dim=dim)
    p.init_index(max_elements=num_elements, ef_construction=800, M=64)
    p.add_items(xb, ids)
    p.set_ef(300)
    p.save_index(bin_out)
    print(f"✅ Saved new index: {bin_out} ({num_elements} items)")


if __name__ == "__main__":
    base_dir = "/data2/jiangsiyu/ATNP_Database/model/"

    build_index(
        pickle_in=os.path.join(base_dir, "herbs_spectra_neg.pickle"),
        pickle_out=os.path.join(base_dir, "herbs_spectra_neg_1000.pickle"),
        bin_out=os.path.join(base_dir, "herbs_index_neg_1000.bin"),
        model_path="model/Ms2Vec_allGNPSnegative.hdf5",
        top_n=1000,
    )

    build_index(
        pickle_in=os.path.join(base_dir, "herbs_spectra_pos.pickle"),
        pickle_out=os.path.join(base_dir, "herbs_spectra_pos_1000.pickle"),
        bin_out=os.path.join(base_dir, "herbs_index_pos_1000.bin"),
        model_path="model/Ms2Vec_allGNPSpositive.hdf5",
        top_n=1000,
    )
