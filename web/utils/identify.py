import pickle
import numpy as np
from gensim.models import Word2Vec
from hnswlib import Index
from matchms.importing import load_from_mgf
from matchms import Spectrum

def find_most_similar_spectrum(s, p, model, references):
    intensities = s.intensities
    if intensities.max() > 1:
        intensities = intensities / intensities.max()

    s_norm = Spectrum(mz=s.mz, intensities=intensities, metadata=s.metadata)

    from spec2vec import SpectrumDocument
    from spec2vec.vector_operations import calc_vector

    query_vector = calc_vector(model, SpectrumDocument(s_norm, n_decimals=2), allowed_missing_percentage=100)
    xq = np.array(query_vector).astype('float32')
    xq /= np.linalg.norm(xq)

    idx, distance = p.knn_query(xq, 1)
    norm_distance = round(distance[0, 0] / 4.0, 2)

    score = 1.0 - norm_distance
    score = round(score, 2)
    ref_spec = np.array(references)[idx[0, 0]]
    ref_spec_index = ref_spec.metadata.get("database_index")
    return ref_spec, score, ref_spec_index

def id_spectrum_list(spectrum_list,
                     hnsw_pos, model_pos, refs_pos,
                     hnsw_neg, model_neg, refs_neg,
                     progress=None):
    res = []
    for s in spectrum_list:
        sn = None
        if "ionmode" in s.metadata.keys():
            if s.metadata["ionmode"] == "negative":
                sn = find_most_similar_spectrum(s, hnsw_neg, model_neg, refs_neg)
            else:
                sn = find_most_similar_spectrum(s, hnsw_pos, model_pos, refs_pos)
        else:
            sn = find_most_similar_spectrum(s, hnsw_pos, model_pos, refs_pos)
        res.append(sn)
    print("hhhres")
    return res

def identify_batch(file_path):
    # 先加载模型和索引（最好启动时全局加载，提高效率）
    MODEL_POS_PATH = '/data2/jiangsiyu/ATNP_Database/model/Ms2Vec_allGNPSpositive.hdf5'
    MODEL_NEG_PATH = '/data2/jiangsiyu/ATNP_Database/model/Ms2Vec_allGNPSnegative.hdf5'
    model_pos = Word2Vec.load(MODEL_POS_PATH)
    model_neg = Word2Vec.load(MODEL_NEG_PATH)

    with open("/data2/jiangsiyu/ATNP_Database/model/references_spectrums_positive.pickle", "rb") as f:
        refs_pos = pickle.load(f)
    with open("/data2/jiangsiyu/ATNP_Database/model/references_spectrums_negative.pickle", "rb") as f:
        refs_neg = pickle.load(f)

    hnsw_pos = Index(space="l2", dim=300)
    hnsw_pos.load_index("/data2/jiangsiyu/ATNP_Database/model/references_index_positive_spec2vec.bin")
    hnsw_pos.set_ef(300)
    hnsw_neg = Index(space="l2", dim=300)
    hnsw_neg.load_index("/data2/jiangsiyu/ATNP_Database/model/references_index_negative_spec2vec.bin")
    hnsw_neg.set_ef(300)

    spectrums = list(load_from_mgf(file_path))
    print(f"✔ 加载上传谱图数：{len(spectrums)}")
    for i, s in enumerate(spectrums):
        print(f"  Spectrum {i}: title={s.get('title')}, ionmode={s.get('ionmode')}, num_peaks={len(s.peaks)}")

    raw_results = id_spectrum_list(spectrums, hnsw_pos, model_pos, refs_pos, hnsw_neg, model_neg, refs_neg)

    from web.models import CompoundLibrary

    results = []
    for query_spec, (ref_spec, score, ref_idx) in zip(spectrums, raw_results):
        compound_obj = None
        if ref_spec:
            std_name = ref_spec.get("standard") or ref_spec.get("name")
            if std_name:
                compound_obj = CompoundLibrary.objects.filter(standard__iexact=std_name).first()

        results.append({
            "query_title": query_spec.metadata.get("title", "Unknown"),
            "score": score,
            "compound_id": compound_obj.id if compound_obj else None,
            "ref_index": ref_idx,
        })

    print(f"identify_batch returned {len(results)} results")
    return results
