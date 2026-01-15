# web/utils/identify.py
# -*- coding: utf-8 -*-
"""
谱图识别与相似谱图匹配（低内存 / 懒加载版）
- spec2vec 模型使用 mmap 只读加载
- HNSW index / refs 按需加载
- 首次使用才占内存，避免 gunicorn 启动即 OOM
"""

import logging
import pickle
import numpy as np
from gensim.models import KeyedVectors
from hnswlib import Index
from spec2vec import SpectrumDocument
from spec2vec.vector_operations import calc_vector
from matchms import Spectrum

logger = logging.getLogger(__name__)
logger.info("identify.py loaded (lazy & low-mem version)")

# =====================================================
# 路径配置
# =====================================================
MODEL_POS_PATH = "/data2/jiangsiyu/ATNP_Database/model/Ms2Vec_allGNPSpositive.hdf5"
MODEL_NEG_PATH = "/data2/jiangsiyu/ATNP_Database/model/Ms2Vec_allGNPSnegative.hdf5"

REFS_POS_PATH = "/data2/jiangsiyu/ATNP_Database/model/herbs_spectra_pos_1.pickle"
REFS_NEG_PATH = "/data2/jiangsiyu/ATNP_Database/model/herbs_spectra_neg_1.pickle"

HNSW_POS_PATH = "/data2/jiangsiyu/ATNP_Database/model/herbs_index_pos_1.bin"
HNSW_NEG_PATH = "/data2/jiangsiyu/ATNP_Database/model/herbs_index_neg_1.bin"

VECTOR_DIM = 300

# =====================================================
# 全局缓存（进程级，只加载一次）
# =====================================================
_models = {}
_indexes = {}
_refs = {}

# =====================================================
# 工具：确保 Spectrum 类型
# =====================================================
def dict_to_spectrum(obj):
    """递归修复 pickle 中的 spectrum 为 matchms.Spectrum"""
    if isinstance(obj, Spectrum):
        return obj

    if isinstance(obj, dict):
        inner = obj
        depth = 0
        while isinstance(inner, dict) and "spectrum" in inner and depth < 5:
            inner = inner["spectrum"]
            depth += 1

        if isinstance(inner, Spectrum):
            return inner

        peaks = inner.get("peaks", []) if isinstance(inner, dict) else []
        metadata = inner.get("metadata", {}) if isinstance(inner, dict) else {}

        if peaks:
            mz, intensities = zip(*peaks)
        else:
            mz, intensities = [], []

        return Spectrum(
            mz=np.asarray(mz, dtype=float),
            intensities=np.asarray(intensities, dtype=float),
            metadata=metadata,
        )

    return obj


# =====================================================
# 懒加载：模型 / index / refs
# =====================================================
def get_model(mode):
    if mode not in _models:
        logger.info(f"[LOAD] spec2vec model ({mode})")
        path = MODEL_POS_PATH if mode == "pos" else MODEL_NEG_PATH
        _models[mode] = KeyedVectors.load(path, mmap="r")  # ⭐ 省内存关键
    return _models[mode]


def get_index(mode):
    if mode not in _indexes:
        logger.info(f"[LOAD] HNSW index ({mode})")
        idx = Index(space="l2", dim=VECTOR_DIM)
        path = HNSW_POS_PATH if mode == "pos" else HNSW_NEG_PATH
        idx.load_index(path)
        idx.set_ef(300)
        _indexes[mode] = idx
    return _indexes[mode]


def get_refs(mode):
    if mode in _refs:
        return _refs[mode]

    logger.info(f"[LOAD] reference spectra ({mode})")
    path = REFS_POS_PATH if mode == "pos" else REFS_NEG_PATH

    with open(path, "rb") as f:
        raw_refs = pickle.load(f)

    fixed_refs = []

    for i, r in enumerate(raw_refs):
        # ---------- 情况 1：r 本身就是 Spectrum ----------
        if isinstance(r, Spectrum):
            fixed_refs.append({
                "spectrum": r,
            })
            continue

        # ---------- 情况 2：r 是 dict ----------
        if isinstance(r, dict):
            spec = r.get("spectrum", r)
            spec = dict_to_spectrum(spec)

            fixed_refs.append({
                "spectrum": spec,
            })
            continue

        # ---------- 兜底 ----------
        logger.warning(f"[REFS] Unsupported ref type at {i}: {type(r)}")

    _refs[mode] = fixed_refs
    logger.info(f"[LOAD] {mode} refs total: {len(fixed_refs)}")

    return _refs[mode]

# =====================================================
# 核心：单谱图相似度搜索
# =====================================================
def find_most_similar_spectrum(spectrum, ionmode="positive", n_decimals=2):
    """
    spectrum: matchms.Spectrum
    ionmode: positive / negative / pos / neg / + / -
    返回 score > 0.6 的去重结果
    """

    mode_map = {
        "positive": "pos", "pos": "pos", "+": "pos",
        "negative": "neg", "neg": "neg", "-": "neg",
    }
    key = str(ionmode).lower() if ionmode else "pos"
    mode = mode_map.get(key, "pos")

    try:
        model = get_model(mode)
        hnsw = get_index(mode)
        references = get_refs(mode)
    except Exception as e:
        logger.error(f"Model/index/refs load failed ({mode}): {e}")
        return []

    # 构建 spec2vec 向量
    sdoc = SpectrumDocument(spectrum, n_decimals=n_decimals)
    vec = calc_vector(model, sdoc, allowed_missing_percentage=100)

    if vec is None or np.linalg.norm(vec) == 0:
        return []

    xq = np.asarray(vec, dtype="float32").reshape(1, -1)
    xq /= np.linalg.norm(xq)

    idxs, distances = hnsw.knn_query(xq, k=500)

    results = []
    for idx, dist in zip(idxs[0], distances[0]):
        score = round(1.0 - dist / 2.0, 5)
        if score <= 0.6:
            continue

        ref_spec = references[idx]["spectrum"]
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

        results.append({
            "herb_id": meta.get("herb_id"),
            "latin_name": meta.get("latin_name"),
            "chinese_name": meta.get("chinese_name"),
            "tissue": meta.get("tissue"),
            "compound_name": meta.get("compound_name", ""),
            "score": score,
            "precursor_mz": precursor_mz,
            "ionmode": meta.get("ionmode") or meta.get("IONMODE"),
            "spectrum_index": idx,
        })

    # 去重（同植物 + 化合物）
    unique = {}
    for r in results:
        k = (r["latin_name"], r["compound_name"])
        if k not in unique or r["score"] > unique[k]["score"]:
            unique[k] = r

    return list(unique.values())


# =====================================================
# 批量接口（保持你原来的 API）
# =====================================================
def identify_spectrums(spectrums):
    all_results = []
    for s in spectrums:
        ionmode = s.metadata.get("ionmode", "positive")
        all_results.extend(find_most_similar_spectrum(s, ionmode=ionmode))
    return all_results
