# web/management/commands/import_standards.py
from django.core.management.base import BaseCommand
from matchms.importing import load_from_mgf
from web.models import CompoundLibrary
import pickle
import re
import json

MAX_CHAR_LENGTH = 255

class Command(BaseCommand):
    help = 'Import a standard MGF (positive or negative) into CompoundLibrary'

    def add_arguments(self, parser):
        parser.add_argument('mgf_path', type=str)
        parser.add_argument('--ionmode', choices=['positive','negative'], required=True)

    def handle(self, mgf_path, ionmode, **opts):
        # -----------------------------------------
        # 手动提取 title -> PEPMASS 映射
        # -----------------------------------------
        title_pepmass_map = {}
        with open(mgf_path, 'r', encoding='utf-8') as f:
            block = []
            for line in f:
                line = line.strip()
                if line == "BEGIN IONS":
                    block = []
                elif line == "END IONS":
                    title = None
                    pepmass_val = None
                    for l in block:
                        if l.startswith("TITLE="):
                            title = l.replace("TITLE=", "").strip()[:MAX_CHAR_LENGTH]
                        if l.startswith("PEPMASS="):
                            pep_str = l.replace("PEPMASS=", "").strip()
                            try:
                                pepmass_val = float(pep_str.split()[0])
                            except:
                                pass
                    if title and pepmass_val:
                        title_pepmass_map[title] = pepmass_val
                else:
                    block.append(line)

        # -----------------------------------------
        # 载入并导入谱图
        # -----------------------------------------
        total, count = 0, 0
        for spec in load_from_mgf(mgf_path):
            total += 1
            meta = spec.metadata
            meta_lower = {k.lower(): v for k, v in meta.items()}

            title_str = meta_lower.get('title') or meta_lower.get('compound_name') or ''
            title_str = title_str[:MAX_CHAR_LENGTH]

            standard_str = meta_lower.get('compound_name') or meta_lower.get('title') or ''
            standard_str = standard_str[:MAX_CHAR_LENGTH]

            pepmass = title_pepmass_map.get(title_str, None)

            print(f"Importing title={title_str}, parsed pepmass={pepmass}")

            obj = CompoundLibrary(
                spectrum_type = 'standard',
                ionmode       = ionmode,
                title         = title_str,
                standard      = standard_str,
                database      = meta_lower.get('database') or 'standard',
                smiles        = meta_lower.get('smiles') or '',
                score         = float(meta_lower.get('score') or 0),
                precursor_mz  = float(meta_lower.get('precursor_mz') or 0),
                rtinseconds   = float(meta_lower.get('rtinseconds') or 0),
                pepmass       = pepmass,
                spectrum_blob = pickle.dumps(spec),
                peaks = ([
                    {"mz": float(m), "int": float(i)}
                    for m, i in zip(spec.peaks.mz, spec.peaks.intensities)
                ])
            )
            obj.save()
            count += 1

        self.stdout.write(self.style.SUCCESS(f"[✓] Imported {count} / {total} spectra from {mgf_path}"))
