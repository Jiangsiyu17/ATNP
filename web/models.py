# web/models.py

import json
import numpy as np
import pickle
from django.db import models
from matchms import Spectrum
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from web.utils.compound_aggregate import normalize_mol
from matchms.filtering import normalize_intensities

class CompoundLibrary(models.Model):
    # ─────────────────────────────
    # 基础标识
    # ─────────────────────────────
    standard_id = models.CharField(max_length=50, null=True, blank=True)
    matched_spectrum_id = models.CharField(max_length=50, null=True, blank=True)

    title = models.CharField(
        max_length=255, blank=True, null=True, db_index=True
    )

    # ─────────────────────────────
    # 结构相关
    # ─────────────────────────────
    smiles = models.TextField(blank=True, null=True)
    inchikey = models.CharField(max_length=27, blank=True, null=True, db_index=True)

    # ⚠️ 只存「最终 Morgan FP」，搜索时不再计算
    morgan_fp = models.BinaryField(blank=True, null=True)

    def get_fingerprint(self):
        """
        从数据库中反序列化 Morgan 指纹
        """
        if not self.morgan_fp:
            return None
        try:
            # Django BinaryField 返回 bytes，可直接用
            return DataStructs.CreateFromBinaryText(self.morgan_fp)
        except Exception:
            return None

    def recalc_fingerprint(self, save=True):
        """
        根据 smiles 重新计算 Morgan FP（normalize 后）
        """
        if not self.smiles:
            self.morgan_fp = None
            return None

        mol = Chem.MolFromSmiles(self.smiles)
        if mol is None:
            self.morgan_fp = None
            return None

        mol = normalize_mol(mol)  # ⭐ 必须

        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=2, nBits=2048
        )

        self.morgan_fp = DataStructs.BitVectToBinaryText(fp)

        if save:
            self.save(update_fields=["morgan_fp"])

        return fp



    # ─────────────────────────────
    # 化合物注释信息
    # ─────────────────────────────
    standard = models.TextField(blank=True, null=True)
    chinese_name = models.CharField(
        max_length=255, blank=True, null=True, db_index=True
    )
    latin_name = models.CharField(
        max_length=255, blank=True, null=True, db_index=True
    )
    tissue = models.CharField(max_length=255, blank=True, null=True)

    precursor_mz = models.FloatField(blank=True, null=True)
    score = models.FloatField(blank=True, null=True)
    database = models.CharField(max_length=255, blank=True, null=True)
    ionmode = models.CharField(max_length=255, blank=True, null=True)

    rtinseconds = models.FloatField(blank=True, null=True)
    pepmass = models.CharField(max_length=255, blank=True, null=True)

    # ─────────────────────────────
    # 谱图类型
    # ─────────────────────────────
    spectrum_type = models.CharField(
        max_length=16,
        choices=[
            ("sample", "sample"),
            ("standard", "standard"),
        ],
        default="sample",
    )

    # ─────────────────────────────
    # 谱图数据
    # ─────────────────────────────
    spectrum_blob = models.BinaryField(blank=True, null=True)
    peaks = models.JSONField(blank=True, null=True)

    # 植物来源（结构搜索不参与）
    plants = models.JSONField(blank=True, null=True)

    class Meta:
        indexes = [
            models.Index(fields=["title", "spectrum_type"]),
        ]

    # ─────────────────────────────
    # matchms Spectrum 还原
    # ─────────────────────────────
    def get_spectrum(self) -> Spectrum | None:
        """
        从 spectrum_blob（优先）或 peaks 还原为 matchms Spectrum
        并确保强度已归一化（spec2vec 必需）
        """
        try:
            # ---------- 1️⃣ 优先使用 spectrum_blob ----------
            if self.spectrum_blob:
                spectrum = pickle.loads(self.spectrum_blob)

                # ⭐ 确保 blob 里的谱图也被归一化
                spectrum = normalize_intensities(spectrum)
                return spectrum

            # ---------- 2️⃣ 从 peaks 构建 ----------
            if self.peaks:
                mz = [p["mz"] for p in self.peaks]
                intensities = [p["int"] for p in self.peaks]

                metadata = {
                    "precursor_mz": self.precursor_mz,
                    "smiles": self.smiles,
                    "name": self.standard,
                    "ionmode": self.ionmode,
                    "compound_id": self.id,
                }

                spectrum = Spectrum(
                    mz=np.array(mz, dtype=float),
                    intensities=np.array(intensities, dtype=float),
                    metadata=metadata,
                )

                # ⭐⭐ 关键：强度归一化 ⭐⭐
                spectrum = normalize_intensities(spectrum)
                return spectrum

        except Exception as e:
            print(f"[get_spectrum error] ID={self.id}: {e}")

        return None
