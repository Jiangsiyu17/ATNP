# -*- coding: utf-8 -*-
from django.core.management.base import BaseCommand
from django.conf import settings
from web.models import CompoundLibrary

from matchms import Spectrum
from matchms.filtering import (
    normalize_intensities,
    select_by_mz,
    select_by_relative_intensity
)

import numpy as np
import pickle
import os
import hnswlib
import gensim

from spec2vec import SpectrumDocument
from spec2vec.vector_operations import calc_vector


MODEL_DIR = settings.BASE_DIR / "model"


IONMODES = {
    "positive": {
        "model": os.path.join(MODEL_DIR, "Ms2Vec_allGNPSpositive.hdf5"),
        "spectra": os.path.join(MODEL_DIR, "standards_spectra_pos.pickle"),
        "index": os.path.join(MODEL_DIR, "standards_index_pos.bin")
    },
    "negative": {
        "model": os.path.join(MODEL_DIR, "Ms2Vec_allGNPSnegative.hdf5"),
        "spectra": os.path.join(MODEL_DIR, "standards_spectra_neg.pickle"),
        "index": os.path.join(MODEL_DIR, "standards_index_neg.bin")
    }
}


def parse_peaks_from_json(peaks):
    """从 JSONField 解析 mz / intensity"""
    if not isinstance(peaks, list):
        return None, None

    mz, intensities = [], []

    for p in peaks:
        if not isinstance(p, dict):
            continue
        if "mz" not in p or "int" not in p:
            continue
        try:
            mz.append(float(p["mz"]))
            intensities.append(float(p["int"]))
        except Exception:
            continue

    if not mz:
        return None, None

    return np.array(mz, dtype=float), np.array(intensities, dtype=float)


class Command(BaseCommand):
    help = "Build standard MS/MS spectra + Spec2Vec HNSW index (positive & negative)"

    def handle(self, *args, **options):

        for ionmode, paths in IONMODES.items():
            self.stdout.write(f"\n=== Processing {ionmode} spectra ===")

            # 1️⃣ 获取该 ionmode 标准品
            qs = (
                CompoundLibrary.objects
                .filter(spectrum_type__iexact="standard")
                .filter(ionmode__iexact=ionmode)
                .filter(peaks__isnull=False)
                .order_by("id")
            )

            total = qs.count()
            parsed, skipped = 0, 0
            spectra = []

            self.stdout.write(f"Total standard objs = {total}")

            # 2️⃣ 构建 Spectrum 列表
            for obj in qs:
                mz, intensities = parse_peaks_from_json(obj.peaks)
                if mz is None:
                    skipped += 1
                    continue

                spectrum = Spectrum(
                    mz=mz,
                    intensities=intensities,
                    metadata={
                        "compound_id": obj.id,
                        "ionmode": obj.ionmode,
                        "precursor_mz": obj.precursor_mz or obj.pepmass,
                    }
                )

                spectra.append(spectrum)
                parsed += 1

            # 保存 spectra.pickle
            with open(paths["spectra"], "wb") as f:
                pickle.dump(spectra, f)

            self.stdout.write(
                self.style.SUCCESS(
                    f"Parsed spectra = {parsed}, Skipped = {skipped}"
                )
            )

            # 3️⃣ 加载 Spec2Vec 模型
            self.stdout.write("Loading spec2vec Word2Vec model...")
            w2v_model = gensim.models.Word2Vec.load(paths["model"])
            kv = w2v_model.wv
            dim = kv.vector_size

            vectors = []
            valid_ids = []

            DECIMALS_CANDIDATES = [3, 2, 1, 0]

            # 4️⃣ 向量化
            for i, spectrum in enumerate(spectra):
                try:
                    # 基本过滤
                    spectrum = select_by_mz(spectrum, mz_from=0, mz_to=1000)
                    spectrum = select_by_relative_intensity(spectrum, intensity_from=0.01)
                    spectrum = normalize_intensities(spectrum)

                    if spectrum.peaks is None or len(spectrum.peaks.mz) < 5:
                        continue

                    vec = None

                    for d in DECIMALS_CANDIDATES:
                        doc = SpectrumDocument(spectrum, n_decimals=d)
                        known = [w for w in doc.words if w in kv.key_to_index]
                        if len(known) < 3:
                            continue
                        vec_tmp = calc_vector(w2v_model, doc, allowed_missing_percentage=5)
                        if vec_tmp is not None:
                            vec = vec_tmp
                            break

                    if vec is None:
                        continue

                    norm = np.linalg.norm(vec)
                    if not np.isfinite(norm) or norm == 0:
                        continue

                    vectors.append(vec)
                    valid_ids.append(i)

                except Exception:
                    continue

            vectors = np.array(vectors, dtype="float32")

            if len(vectors) == 0:
                self.stderr.write(f"❌ No valid vectors generated for {ionmode}")
                continue

            # 单位化
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

            self.stdout.write(
                self.style.SUCCESS(
                    f"✅ {ionmode} vectors: {vectors.shape}"
                )
            )

            # 5️⃣ 构建 HNSW index
            index = hnswlib.Index(space="cosine", dim=dim)
            index.init_index(max_elements=len(vectors), ef_construction=400, M=64)
            index.add_items(vectors, np.arange(len(vectors)))
            index.set_ef(300)
            index.save_index(paths["index"])

            self.stdout.write(
                self.style.SUCCESS(
                    f"✅ Built {ionmode} MS/MS index: {paths['index']}"
                )
            )

        self.stdout.write(self.style.SUCCESS("\nAll done!"))
