# web/models.py

import json
import numpy as np
from django.db import models
from matchms import Spectrum
import pickle


class CompoundLibrary(models.Model):
    # ─── 新增 title，用于匹配 ───────────────────────
    title = models.CharField(max_length=255, blank=True, null=True, db_index=True)

    # ─── SMILES 字段，去掉唯一约束和索引 ───
    smiles = models.TextField(blank=True, null=True)

    # ─── 基本字段 ───────────────────────────────
    standard = models.CharField(max_length=255, blank=True, null=True, db_index=True)
    chinese_name = models.CharField(max_length=255, blank=True, null=True, db_index=True)
    latin_name = models.CharField(max_length=255, blank=True, null=True, db_index=True)
    tissue = models.CharField(max_length=255, blank=True, null=True)

    precursor_mz = models.FloatField(blank=True, null=True)
    score = models.FloatField(blank=True, null=True)
    database = models.CharField(max_length=255, blank=True, null=True)
    ionmode = models.CharField(max_length=255, blank=True, null=True)

    rtinseconds = models.FloatField(blank=True, null=True)
    pepmass = models.CharField(max_length=255, blank=True, null=True)

    # ─── 谱图类型 ──────────────────────────────
    spectrum_type = models.CharField(
        max_length=16,
        choices=[('sample', 'sample'), ('standard', 'standard')],
        default='sample'
    )

    # ─── 谱图数据 ──────────────────────────────
    spectrum_blob = models.BinaryField(blank=True, null=True)
    peaks = models.JSONField(blank=True, null=True)

    class Meta:
        indexes = [
            models.Index(fields=['title', 'spectrum_type']),  # 用于标品快速匹配
        ]

    def get_spectrum(self) -> Spectrum | None:
        """
        从 spectrum_blob（优先）或 peaks 字段还原为 matchms Spectrum 对象。
        """
        try:
            if self.spectrum_blob:
                return pickle.loads(self.spectrum_blob)
            if self.peaks:
                mz = [p['mz'] for p in self.peaks]
                ints = [p['int'] for p in self.peaks]
                metadata = {
                    "precursor_mz": self.precursor_mz,
                    "smiles": self.smiles,
                    "name": self.standard,
                    "ionmode": self.ionmode,
                }
                return Spectrum(mz=np.array(mz), intensities=np.array(ints), metadata=metadata)
        except Exception as e:
            print(f"get_spectrum error: {e}")
        return None

