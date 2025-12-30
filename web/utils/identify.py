# web/utils/identify.py
# -*- coding: utf-8 -*-
"""
谱图识别与相似谱图匹配
支持正负离子谱图库，自动修复 pickle 中的 spectrum 类型
"""

import logging
import pickle
import numpy as np
from gensim.models import Word2Vec
from hnswlib import Index
from spec2vec import SpectrumDocument
from spec2vec.vector_operations import calc_vector
from matchms import Spectrum
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

# ---------- 工具函数 ----------
def dict_to_spectrum(obj):
    """递归修复谱图对象为 Spectrum"""
    if isinstance(obj, Spectrum):
        return obj
    if isinstance(obj, dict):
        # 连续展开多层嵌套
        inner = obj
        depth = 0
        while isinstance(inner, dict) and "spectrum" in inner and depth < 5:
            inner = inner["spectrum"]
            depth += 1
        if isinstance(inner, Spectrum):
            return inner
        peaks = inner.get("peaks", [])
        metadata = inner.get("metadata", {})
        if peaks:
            mz, intensities = zip(*peaks)
        else:
            mz, intensities = [], []
        return Spectrum(mz=np.array(mz), intensities=np.array(intensities), metadata=metadata)
    return obj

# ---------- 模型与索引加载 ----------
def load_models_and_indexes():
    """懒加载模型、索引和谱图，确保 refs 中 spectrum 为 Spectrum"""
    global _models, _indexes, _refs

    def fix_refs(ref_list):
        """把每个条目的 spectrum 字段保证是 Spectrum 对象"""
        for i, r in enumerate(ref_list):
            spec = r.get("spectrum")
            if not isinstance(spec, Spectrum):
                r["spectrum"] = dict_to_spectrum(spec)
                # 再次校验
                if not isinstance(r["spectrum"], Spectrum):
                    raise TypeError(f"Ref entry {i} failed to convert to Spectrum: {type(spec)}")
        return ref_list

    # ---------- 正离子 ----------
    if "pos" not in _models:
        try:
            logger.info("Loading POS model and index...")
            _models["pos"] = Word2Vec.load(MODEL_POS_PATH)
            _indexes["pos"] = Index(space="l2", dim=VECTOR_DIM)
            _indexes["pos"].load_index(HNSW_POS_PATH)
            _indexes["pos"].set_ef(300)
            with open(REFS_POS_PATH, "rb") as f:
                refs = pickle.load(f)
            _refs["pos"] = fix_refs(refs)
            logger.info(f"POS refs loaded, total: {len(_refs['pos'])}")
        except Exception as e:
            logger.warning(f"Failed to load POS model/index/refs: {e}")

    # ---------- 负离子 ----------
    if "neg" not in _models:
        try:
            logger.info("Loading NEG model and index...")
            _models["neg"] = Word2Vec.load(MODEL_NEG_PATH)
            _indexes["neg"] = Index(space="l2", dim=VECTOR_DIM)
            _indexes["neg"].load_index(HNSW_NEG_PATH)
            _indexes["neg"].set_ef(300)
            with open(REFS_NEG_PATH, "rb") as f:
                refs = pickle.load(f)
            _refs["neg"] = fix_refs(refs)
            logger.info(f"NEG refs loaded, total: {len(_refs['neg'])}")
        except Exception as e:
            logger.warning(f"Failed to load NEG model/index/refs: {e}")

# ---------- 谱图比对 ----------
def find_most_similar_spectrum(spectrum, ionmode="positive", n_decimals=2):
    """
    spectrum: matchms.Spectrum 对象
    ionmode: 'positive', 'negative', 'pos', 'neg', '+', '-'
    返回所有相似谱图信息，自动去重（同植物+化合物）并只保留 score>0.6
    """
    load_models_and_indexes()

    mode_map = {
        "positive": "pos", "pos": "pos", "+": "pos",
        "negative": "neg", "neg": "neg", "-": "neg",
    }
    key = str(ionmode).lower() if ionmode else "pos"
    mode = mode_map.get(key)
    if mode is None or mode not in _refs or mode not in _models or mode not in _indexes:
        logger.warning(f"Mode '{key}' not loaded or unavailable, skipping spectrum.")
        return []

    model = _models[mode]
    hnsw = _indexes[mode]
    references = _refs[mode]

    # 构建 SpectrumDocument 并计算向量
    sdoc = SpectrumDocument(spectrum, n_decimals=n_decimals)
    vec = calc_vector(model, sdoc, allowed_missing_percentage=100)

    if vec is None or np.linalg.norm(vec) == 0:
        logger.warning(f"Spectrum {spectrum.metadata.get('compound_name','?')} produced empty vector. Skipping.")
        return []

    xq = np.array(vec, dtype="float32").reshape(1, -1)
    xq /= np.linalg.norm(xq)

    # knn_query
    res = hnsw.knn_query(xq, k=500)
    idxs, distances = res if isinstance(res, tuple) else (np.array([[res]]), np.array([[0.0]]))

    temp_results = []
    for idx, dist in zip(idxs[0], distances[0]):
        score = round(1.0 - dist / 2.0, 5)
        if score <= 0.6:
            continue

        ref_entry = references[idx]
        ref_spec = ref_entry["spectrum"]  # 此时已经是 Spectrum 对象


        meta = ref_spec.metadata or {}

        precursor_mz = (
            meta.get("precursor_mz")
            or meta.get("PRECURSOR_MZ")
            or meta.get("pepmass")
            or meta.get("PEPMASS")
        )

        try:
            precursor_mz = round(float(precursor_mz), 4) if precursor_mz else None
        except Exception:
            precursor_mz = None

        ionmode = (
            meta.get("ionmode")
            or meta.get("IONMODE")
            or meta.get("polarity")
            or "-"
        )

        temp_results.append({
            "herb_id": meta.get("herb_id"),
            "latin_name": meta.get("latin_name"),
            "chinese_name": meta.get("chinese_name"),
            "tissue": meta.get("tissue"),
            "score": score,
            "compound_name": meta.get("compound_name", ""),
            "spectrum_index": idx,

            # ✅ 关键新增（样本谱图）
            "precursor_mz": precursor_mz,
            "ionmode": ionmode,
        })


    # 去重
    unique_dict = {}
    for r in temp_results:
        key = (r["latin_name"], r.get("compound_name", ""))
        if key not in unique_dict or r["score"] > unique_dict[key]["score"]:
            unique_dict[key] = r

    return list(unique_dict.values())

# ---------- 批量鉴定 ----------
def identify_spectrums(spectrums):
    """批量鉴定谱图列表"""
    all_results = []
    for s in spectrums:
        ionmode = s.metadata.get("ionmode", "positive")
        res = find_most_similar_spectrum(s, ionmode=ionmode)
        all_results.extend(res)
    return all_results
