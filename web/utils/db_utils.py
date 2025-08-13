# -*- coding: utf-8 -*-
# 处理导入的谱图，生成bin和pkl文件

import numpy as np
import pickle
import hnswlib
import gensim
from tqdm import tqdm
from pathlib import Path
from matchms import Spectrum
from matchms.importing import load_from_mgf, load_from_msp
from matchms.filtering import normalize_intensities  # 归一化峰强度

from spec2vec import SpectrumDocument
from spec2vec.vector_operations import calc_vector
from web.models import CompoundLibrary

import sys
import os
import django
# 设置项目根目录和 Django 配置模块
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ATNP.settings")

django.setup()


def load_spectrum_file(file_path: str | Path):
    """
    根据后缀读取 .mgf 或 .msp，返回 Spectrum 列表。
    自动补充缺失的 compound_name 和 precursor_mz。
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".mgf":
        raw_spectra = load_from_mgf(path)
    elif ext == ".msp":
        raw_spectra = load_from_msp(path)
    else:
        raise ValueError(f"暂不支持 {ext} 格式（仅支持 .mgf / .msp）")

    processed = []
    for spec in raw_spectra:
        spec_new = Spectrum(mz=spec.mz, intensities=spec.intensities, metadata=spec.metadata)

        # 自动补充 compound_name
        if spec_new.get("compound_name") is None:
            spec_new.set("compound_name", f"Unknown_{len(processed)}")

        # ✅ 自动从 pepmass 设置 precursor_mz
        if spec_new.get("precursor_mz") is None:
            pepmass = spec_new.get("pepmass")
            if isinstance(pepmass, (list, tuple)) and len(pepmass) > 0:
                spec_new.set("precursor_mz", float(pepmass[0]))
            elif isinstance(pepmass, str):
                spec_new.set("precursor_mz", float(pepmass.strip().split()[0]))

        processed.append(spec_new)

    return processed


def calc_ms2vec_vector(spec, model):
    spec_norm = normalize_intensities(spec)
    if spec_norm.intensities is None or len(spec_norm.intensities) == 0:
        return None
    doc = SpectrumDocument(spec_norm, n_decimals=2)
    vec = calc_vector(model, doc, allowed_missing_percentage=100)
    return vec


def process_positive_spectrums(pos_mgf_path, pos_model_path, output_pickle, output_index):
    spectrums = load_spectrum_file(pos_mgf_path)

    for idx, s in enumerate(spectrums):
        s.set("database_index", idx)
        s.set("ionmode", "positive")

        std_name = s.metadata.get("standard") or s.metadata.get("compound_name") or s.get("title")
        obj = CompoundLibrary.objects.filter(standard=std_name).first()
        pk = obj.pk if obj else None
        s.set("compound_pk", pk)

    with open(output_pickle, "wb") as f:
        pickle.dump(spectrums, f)
    print(f"→ Dumped {len(spectrums)} positive spectrums to {output_pickle}")

    pos_model = gensim.models.Word2Vec.load(pos_model_path)
    reference_vector = []
    for s in tqdm(spectrums, desc="positive vector calc"):
        vec = calc_ms2vec_vector(s, pos_model)
        if vec is not None:
            reference_vector.append(vec)

    xb = np.array(reference_vector).astype('float32')
    xb /= np.linalg.norm(xb, axis=1, keepdims=True)

    dim = xb.shape[1]
    p = hnswlib.Index(space='l2', dim=dim)
    p.init_index(max_elements=len(xb), ef_construction=800, M=64)
    p.add_items(xb, np.arange(len(xb)))
    p.set_ef(300)
    p.save_index(output_index)


def process_negative_spectrums(neg_mgf_path, neg_model_path, output_pickle, output_index):
    spectrums = load_spectrum_file(neg_mgf_path)

    for idx, s in enumerate(spectrums):
        s.set("database_index", idx)
        s.set("ionmode", "negative")

    with open(output_pickle, "wb") as f:
        pickle.dump(spectrums, f)
    print(f"→ Dumped {len(spectrums)} negative spectrums to {output_pickle}")

    neg_model = gensim.models.Word2Vec.load(neg_model_path)
    reference_vector = []
    for s in tqdm(spectrums, desc="negative vector calc"):
        vec = calc_ms2vec_vector(s, neg_model)
        if vec is not None:
            reference_vector.append(vec)

    xb = np.array(reference_vector).astype('float32')
    xb /= np.linalg.norm(xb, axis=1, keepdims=True)

    dim = xb.shape[1]
    p = hnswlib.Index(space='l2', dim=dim)
    p.init_index(max_elements=len(xb), ef_construction=800, M=64)
    p.add_items(xb, np.arange(len(xb)))
    p.set_ef(300)
    p.save_index(output_index)

def load_plant_spectra_from_folder(folder: str, ionmode: str = "positive"):
    import glob
    from matchms.importing import load_from_mgf
    from matchms.filtering import default_filters

    all_spectra = []
    mgf_files = glob.glob(os.path.join(folder, "*.mgf"))
    for mgf_file in mgf_files:
        spectra = load_from_mgf(mgf_file)
        for spectrum in spectra:
            if spectrum is None:
                continue
            spectrum = default_filters(spectrum)
            meta = spectrum.metadata

            # 不用 "spectrum_type"，改成 "source_type"
            meta["source_type"] = "plant"
            meta["ionmode"] = ionmode
            meta["source_file"] = os.path.basename(mgf_file)

            # 统一字段名
            for key in ["LATIN_NAME", "CHINESE_NAME", "TISSUE"]:
                if key in meta:
                    meta[key.lower()] = meta.pop(key)

            spectrum.metadata = meta
            all_spectra.append(spectrum)
    return all_spectra