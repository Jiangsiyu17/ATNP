# web/management/commands/import_mgf.py

from django.core.management.base import BaseCommand
from matchms.importing import load_from_mgf
from web.models import CompoundLibrary
import pickle
from tqdm import tqdm
import logging
import json

class Command(BaseCommand):
    help = "Import a SAMPLE-library MGF into CompoundLibrary"

    def add_arguments(self, parser):
        parser.add_argument("mgf_path", type=str)

    def handle(self, mgf_path, **opts):
        logging.getLogger("matchms").setLevel(logging.ERROR)  # 关闭警告日志

        spectra = list(load_from_mgf(mgf_path))  # 先加载全部谱图以显示进度
        for spec in tqdm(spectra, desc="Importing MGF"):
            meta = {k.lower(): v for k, v in spec.metadata.items()}

            compound_name = (
                meta.get("standard")
                or meta.get("name")
                or meta.get("title")
                or "unknown"
            )

            obj = CompoundLibrary(
                title         = meta.get("title") or compound_name,
                standard      = meta.get('standard', '') or meta.get('name', '') or compound_name,
                spectrum_type = "sample",
                smiles        = meta.get("smiles") or "",
                database      = meta.get("database") or "sample",
                ionmode       = meta.get("ionmode") or "",
                score         = float(meta.get("score") or 0),
                chinese_name  = meta.get("chinese_name") or "",
                latin_name    = meta.get("latin_name") or "",
                tissue        = meta.get("tissue") or "",
                rtinseconds   = float(meta.get("retention_time") or 0),
                pepmass       = meta.get("pepmass") or meta.get("precursor_mz") or "", 
                spectrum_blob = pickle.dumps(spec),
                peaks = [
                    {"mz": float(m), "int": float(i)}
                    for m, i in zip(spec.peaks.mz, spec.peaks.intensities)
                ]
            )
            obj.save()

        self.stdout.write(self.style.SUCCESS("✓ Sample MGF imported"))
