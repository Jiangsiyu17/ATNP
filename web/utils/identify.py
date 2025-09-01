# web/utils/identify.py

import logging
import pickle
import numpy as np
from gensim.models import Word2Vec
from hnswlib import Index
from spec2vec import SpectrumDocument
from spec2vec.vector_operations import calc_vector
from web.models import CompoundLibrary

logger = logging.getLogger(__name__)
logger.info("identify.py loaded")

# ---------- 全局模型路径 ----------
MODEL_POS_PATH = "/data2/jiangsiyu/ATNP_Database/model/Ms2Vec_allGNPSpositive.hdf5"
MODEL_NEG_PATH = "/data2/jiangsiyu/ATNP_Database/model/Ms2Vec_allGNPSnegative.hdf5"
REFS_POS_PATH = "/data2/jiangsiyu/ATNP_Database/model/herbs_spectra_pos.pickle"
REFS_NEG_PATH = "/data2/jiangsiyu/ATNP_Database/model/herbs_spectra_neg.pickle"
HNSW_POS_PATH = "/data2/jiangsiyu/ATNP_Database/model/herbs_index_pos.bin"
HNSW_NEG_PATH = "/data2/jiangsiyu/ATNP_Database/model/herbs_index_neg.bin"

VECTOR_DIM = 300

# ---------- 全局缓存 ----------
_models = {}
_indexes = {}
_refs = {}

def load_models_and_indexes():
    """懒加载模型、索引和谱图"""
    global _models, _indexes, _refs

    if "pos" not in _models:
        logger.info("Loading POS model and index...")
        _models["pos"] = Word2Vec.load(MODEL_POS_PATH)
        _indexes["pos"] = Index(space="l2", dim=VECTOR_DIM)
        _indexes["pos"].load_index(HNSW_POS_PATH)
        _indexes["pos"].set_ef(300)
        with open(REFS_POS_PATH, "rb") as f:
            _refs["pos"] = pickle.load(f)

    if "neg" not in _models:
        logger.info("Loading NEG model and index...")
        _models["neg"] = Word2Vec.load(MODEL_NEG_PATH)
        _indexes["neg"] = Index(space="l2", dim=VECTOR_DIM)
        _indexes["neg"].load_index(HNSW_NEG_PATH)
        _indexes["neg"].set_ef(300)
        with open(REFS_NEG_PATH, "rb") as f:
            _refs["neg"] = pickle.load(f)

def find_most_similar_spectrum(spectrum, ionmode="positive"):
    """
    spectrum: matchms.Spectrum 对象
    ionmode: 'positive' 或 'negative'
    返回所有相似谱图信息，自动去重（同植物+化合物）并只保留 score>0.6
    """
    load_models_and_indexes()
    
    mode_map = {
        "positive": "pos",
        "pos": "pos",
        "+": "pos",
        "negative": "neg",
        "neg": "neg",
        "-": "neg",
    }

    key = ionmode.lower() if ionmode else "pos"
    if key not in mode_map:
        raise ValueError(f"Unsupported ionmode: {ionmode}")

    mode = mode_map[key]

    model = _models[mode]
    hnsw = _indexes[mode]
    references = _refs[mode]

    # 构建 SpectrumDocument 并计算向量
    sdoc = SpectrumDocument(spectrum, n_decimals=2)
    vec = calc_vector(model, sdoc, allowed_missing_percentage=100)
    xq = np.array(vec, dtype="float32").reshape(1, -1)
    xq /= np.linalg.norm(xq)

    # knn_query
    res = hnsw.knn_query(xq, k=500)
    if isinstance(res, tuple):
        idxs, distances = res
    else:
        idxs = np.array([[res]])
        distances = np.array([[0.0]])

    temp_results = []
    for idx, dist in zip(idxs[0], distances[0]):
        score = round(1.0 - dist / 4.0, 4)
        if score <= 0.6:  # 只保留 score>0.6
            continue
        ref_spec = references[idx]
        temp_results.append({
            "latin_name": ref_spec.metadata.get("latin_name"),
            "chinese_name": ref_spec.metadata.get("chinese_name"),
            "tissue": ref_spec.metadata.get("tissue"),
            "score": score,
            "compound_name": ref_spec.metadata.get("compound_name")  # 如果有
        })

    # 去重：同一植物 + 同一化合物只保留得分最高的
    unique_dict = {}
    for r in temp_results:
        key = (r["latin_name"], r.get("compound_name", ""))
        if key not in unique_dict or r["score"] > unique_dict[key]["score"]:
            unique_dict[key] = r

    return list(unique_dict.values())


def identify_spectrums(spectrums):
    """批量鉴定谱图列表"""
    all_results = []
    for s in spectrums:
        ionmode = s.metadata.get("ionmode", "positive")
        res = find_most_similar_spectrum(s, ionmode=ionmode)
        all_results.extend(res)
    return all_results
