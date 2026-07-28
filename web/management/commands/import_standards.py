# web/management/commands/import_standards.py

from django.core.management.base import BaseCommand
from django.db import transaction, close_old_connections
from matchms.importing import load_from_mgf
from web.models import CompoundLibrary

import pickle
import traceback
import time

from rdkit import Chem
from rdkit.Chem import Descriptors


MAX_CHAR_LENGTH = 255

# ⭐ 稳定性核心：降低 batch
BATCH_SIZE = 50

# retry参数
MAX_RETRY = 3
RETRY_SLEEP = 2


# =========================
# MW计算函数
# =========================
def calc_mw(smiles):
    if not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return round(Descriptors.MolWt(mol), 4)
    except Exception:
        return None


class Command(BaseCommand):
    help = 'Stable import MGF into CompoundLibrary (production safe version)'

    def add_arguments(self, parser):
        parser.add_argument('mgf_path', type=str)
        parser.add_argument('--ionmode', choices=['positive', 'negative'], required=True)

    def handle(self, mgf_path, ionmode, **options):

        batch = []
        total = 0

        for spec in load_from_mgf(mgf_path):
            total += 1

            # ⭐ 防连接泄漏（非常关键）
            if total % 100 == 0:
                close_old_connections()

            meta = spec.metadata
            meta = {k.lower(): v for k, v in meta.items()}

            title = (meta.get("title") or meta.get("compound_name") or "")[:MAX_CHAR_LENGTH]
            standard = (meta.get("compound_name") or meta.get("title") or "")[:MAX_CHAR_LENGTH]

            # standard_id
            raw_id = meta.get("standard_id")
            try:
                standard_id = str(int(float(raw_id))) if raw_id else None
            except:
                standard_id = str(raw_id) if raw_id else None

            # precursor / pepmass
            try:
                pepmass = float(meta.get("pepmass") or 0)
            except:
                pepmass = 0.0

            try:
                precursor = float(meta.get("precursor_mz") or 0)
            except:
                precursor = 0.0

            if precursor == 0:
                precursor = pepmass

            smiles = meta.get("smiles") or ""
            mw = calc_mw(smiles)

            obj = CompoundLibrary(
                spectrum_type='standard',
                ionmode=ionmode,
                title=title,
                standard=standard,
                database=meta.get("database") or "standard",
                smiles=smiles,
                mw=mw,
                inchikey=meta.get("inchikey") or meta.get("inchi_key"),
                score=float(meta.get("score") or 0),
                precursor_mz=precursor,
                rtinseconds=float(meta.get("rtinseconds") or 0),
                pepmass=pepmass,
                standard_id=standard_id,
                antitumor=(str(meta.get("antitumor", "")).upper() == "TRUE"),
                spectrum_blob=pickle.dumps(spec),
                peaks=[
                    {"mz": float(m), "int": float(i)}
                    for m, i in zip(spec.peaks.mz, spec.peaks.intensities)
                ]
            )

            batch.append(obj)

            # =========================
            # ⭐ batch insert
            # =========================
            if len(batch) >= BATCH_SIZE:
                self._safe_bulk_insert(batch)
                batch = []

        # 最后一批
        if batch:
            self._safe_bulk_insert(batch)

        self.stdout.write(self.style.SUCCESS(f"[DONE] Imported {total} spectra"))

    # =========================
    # ⭐ 稳定插入核心函数
    # =========================
    def _safe_bulk_insert(self, batch):

        for attempt in range(MAX_RETRY):

            try:
                with transaction.atomic():
                    CompoundLibrary.objects.bulk_create(
                        batch,
                        batch_size=BATCH_SIZE,
                        ignore_conflicts=True
                    )
                return

            except Exception as e:
                self.stderr.write(f"[WARN] bulk insert failed (try {attempt+1}): {e}")

                # 失败退避
                time.sleep(RETRY_SLEEP * (attempt + 1))

        # =========================
        # fallback：逐条插入（但不save爆炸）
        # =========================
        self.stderr.write("[FALLBACK] inserting one by one...")

        for obj in batch:
            try:
                CompoundLibrary.objects.create(
                    **{
                        "spectrum_type": obj.spectrum_type,
                        "ionmode": obj.ionmode,
                        "title": obj.title,
                        "standard": obj.standard,
                        "database": obj.database,
                        "smiles": obj.smiles,
                        "mw": obj.mw,
                        "inchikey": obj.inchikey,
                        "score": obj.score,
                        "precursor_mz": obj.precursor_mz,
                        "rtinseconds": obj.rtinseconds,
                        "pepmass": obj.pepmass,
                        "standard_id": obj.standard_id,
                        "antitumor": obj.antitumor,
                        "spectrum_blob": obj.spectrum_blob,
                        "peaks": obj.peaks,
                    }
                )
            except Exception as e:
                self.stderr.write(f"[ERROR] single insert failed: {e}")