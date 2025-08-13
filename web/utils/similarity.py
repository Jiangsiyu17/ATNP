import gensim
import hnswlib
import pickle
import os
import numpy as np
from .db_utils import calc_ms2vec_vector as spectrum_to_vector
from web.models import CompoundLibrary

MODEL_DIR = "/data2/jiangsiyu/ATNP_Database/model"

# 谱图库（Spectrum对象列表）
POS_SPECTRA_PATH = os.path.join(MODEL_DIR, "herbs_spectra_pos.pickle")
NEG_SPECTRA_PATH = os.path.join(MODEL_DIR, "herbs_spectra_neg.pickle")

# 索引文件
POS_INDEX_PATH = os.path.join(MODEL_DIR, "herbs_index_pos.bin")
NEG_INDEX_PATH = os.path.join(MODEL_DIR, "herbs_index_neg.bin")

# spec2vec模型文件（示例路径，请根据实际改）
POS_SPEC2VEC_PATH = os.path.join(MODEL_DIR, "Ms2Vec_allGNPSpositive.hdf5")
NEG_SPEC2VEC_PATH = os.path.join(MODEL_DIR, "Ms2Vec_allGNPSnegative.hdf5")

_loaded_models = {}
_loaded_indexes = {}
_loaded_spectra = {}

VECTOR_DIM = 300  # 确认spec2vec模型的向量维度

def load_spec2vec_model(ionmode):
    if ionmode not in _loaded_models:
        if ionmode.lower() == "positive":
            _loaded_models[ionmode] = gensim.models.KeyedVectors.load(POS_SPEC2VEC_PATH)
        else:
            _loaded_models[ionmode] = gensim.models.KeyedVectors.load(NEG_SPEC2VEC_PATH)
    return _loaded_models[ionmode]

def load_index(ionmode):
    if ionmode not in _loaded_indexes:
        index = hnswlib.Index(space="cosine", dim=VECTOR_DIM)
        if ionmode.lower() == "positive":
            index.load_index(POS_INDEX_PATH)
        else:
            index.load_index(NEG_INDEX_PATH)
        _loaded_indexes[ionmode] = index
    return _loaded_indexes[ionmode]

def load_spectra(ionmode):
    if ionmode not in _loaded_spectra:
        if ionmode.lower() == "positive":
            with open(POS_SPECTRA_PATH, "rb") as f:
                _loaded_spectra[ionmode] = pickle.load(f)
        else:
            with open(NEG_SPECTRA_PATH, "rb") as f:
                _loaded_spectra[ionmode] = pickle.load(f)
    return _loaded_spectra[ionmode]

def load_model_and_index(ionmode):
    model = load_spec2vec_model(ionmode)
    index = load_index(ionmode)
    return model, index

def generate_spectrum_comparison(compound, top_k=20, score_threshold=None):
    if compound is None:
        return []

    spec = compound.get_spectrum()
    if spec is None:
        return []

    ionmode = compound.ionmode or "positive"
    model, index = load_model_and_index(ionmode)

    query_vector = spectrum_to_vector(spec, model)
    if query_vector is None:
        return []

    labels, distances = index.knn_query(query_vector, k=top_k)
    similarities = 1 - distances[0]

    results = []
    for label, score in zip(labels[0], similarities):
        # 如果 score_threshold 是 None，就不过滤
        if score_threshold is None or score >= score_threshold:
            try:
                sample_obj = CompoundLibrary.objects.get(pk=int(label))
                results.append({
                    "sample": sample_obj,
                    "score": float(score)
                })
            except CompoundLibrary.DoesNotExist:
                continue
    return results

