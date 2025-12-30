from django.core.management.base import BaseCommand
from web.models import CompoundLibrary
from matchms import Spectrum
from matchms.filtering import normalize_intensities

import numpy as np
import pickle
import os
import hnswlib

import gensim
from spec2vec import SpectrumDocument
from spec2vec.vector_operations import calc_vector


MODEL_DIR = "/data2/jiangsiyu/ATNP_Database/model"

SPECTRA_PATH = os.path.join(MODEL_DIR, "standards_spectra.pickle")
INDEX_PATH   = os.path.join(MODEL_DIR, "standards_index.bin")

# ✅ 直接复用你已有的 GNPS 正离子模型
MODEL_PATH  = os.path.join(MODEL_DIR, "Ms2Vec_allGNPSpositive.hdf5")


def parse_peaks_from_json(peaks):
    """
    peaks: JSONField
    [
        {"mz": float, "int": float},
        ...
    ]
    """
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
    help = "Build standard MS/MS spectra + Spec2Vec HNSW index"

    def handle(self, *args, **options):

        qs = (
            CompoundLibrary.objects
            .filter(spectrum_type__iexact="standard")
            .filter(peaks__isnull=False)
            .order_by("id")   # ⚠️ 顺序必须固定
        )

        total = qs.count()
        parsed, skipped = 0, 0
        spectra = []

        self.stdout.write(f"Total standard objs = {total}")

        # ======================================================
        # 1️⃣ 构建 Spectrum 列表
        # ======================================================
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

            spectrum = normalize_intensities(spectrum)

            spectra.append(spectrum)
            parsed += 1

        # ======================================================
        # 2️⃣ 保存 spectra.pickle
        # ======================================================
        with open(SPECTRA_PATH, "wb") as f:
            pickle.dump(spectra, f)

        self.stdout.write(
            self.style.SUCCESS(
                f"Parsed spectra = {parsed}, Skipped = {skipped}"
            )
        )

        # ======================================================
        # 3️⃣ 计算 Spec2Vec 向量（与你植物库一致）
        # ======================================================
        self.stdout.write("Loading spec2vec Word2Vec model...")

        w2v_model = gensim.models.Word2Vec.load(MODEL_PATH)
        kv = w2v_model.wv
        dim = kv.vector_size

        vectors = []
        valid_ids = []

        for i, spectrum in enumerate(spectra):
            try:
                doc = SpectrumDocument(spectrum, n_decimals=3)
                vec = calc_vector(
                    w2v_model,   # ⚠️ model 在前
                    doc,
                    allowed_missing_percentage=100
                )
                if vec is None:
                    continue

                vectors.append(vec)
                valid_ids.append(i)

            except Exception:
                continue

            vectors = np.array(vectors, dtype="float32")

            if len(vectors) == 0:
                self.stderr.write("❌ No valid vectors generated.")
                return

            # ======================================================
            # 🔥 过滤 zero / NaN 向量（必须）
            # ======================================================
            norms = np.linalg.norm(vectors, axis=1)

            valid_mask = np.isfinite(norms) & (norms > 0)

            vectors = vectors[valid_mask]
            valid_ids = [valid_ids[i] for i in range(len(valid_ids)) if valid_mask[i]]

            if len(vectors) == 0:
                self.stderr.write("❌ All vectors were zero or NaN after filtering.")
                return

            # ======================================================
            # 🔑 单位化（和植物库完全一致）
            # ======================================================
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

            self.stdout.write(
                self.style.SUCCESS(
                    f"Computed vectors after filtering: {vectors.shape}"
                )
            )


        # ======================================================
        # 4️⃣ 构建 HNSW index（生成 bin）
        # ======================================================
        index = hnswlib.Index(space="cosine", dim=dim)
        index.init_index(
            max_elements=len(vectors),
            ef_construction=400,
            M=64
        )

        index.add_items(vectors, np.arange(len(vectors)))
        index.set_ef(300)
        index.save_index(INDEX_PATH)

        self.stdout.write(
            self.style.SUCCESS(
                f"✅ Built standard MS/MS index: {INDEX_PATH}"
            )
        )
