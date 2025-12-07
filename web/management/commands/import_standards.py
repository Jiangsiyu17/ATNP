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
        parser.add_argument('--ionmode', choices=['positive', 'negative'], required=True)

    def handle(self, mgf_path, ionmode, **opts):

        # ==========================================================
        #  1) 手动解析 MGF，提取 title → pepmass（支持两种情况）
        # ==========================================================
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
                    precursor_mz_val = None

                    for l in block:
                        if l.startswith("TITLE="):
                            title = l.replace("TITLE=", "").strip()[:MAX_CHAR_LENGTH]

                        if l.startswith("PEPMASS="):
                            try:
                                pepmass_val = float(l.split("=", 1)[1].split()[0])
                            except:
                                pass

                        if l.startswith("PRECURSOR_MZ="):
                            try:
                                precursor_mz_val = float(l.split("=", 1)[1].split()[0])
                            except:
                                pass

                    # ---- 优先 PEPMASS，其次用 PRECURSOR_MZ ----
                    final_pepmass = pepmass_val or precursor_mz_val

                    if title and final_pepmass:
                        title_pepmass_map[title] = final_pepmass

                else:
                    block.append(line)

        # ==========================================================
        #  2) preload 标品谱图（matchms）
        # ==========================================================
        total, count = 0, 0
        for spec in load_from_mgf(mgf_path):
            total += 1

            meta = spec.metadata
            meta_lower = {k.lower(): v for k, v in meta.items()}

            title_str = meta_lower.get('title') or meta_lower.get('compound_name') or ''
            title_str = title_str[:MAX_CHAR_LENGTH]

            standard_str = meta_lower.get('compound_name') or meta_lower.get('title') or ''
            standard_str = standard_str[:MAX_CHAR_LENGTH]

            # ======================================================
            #  3) 规范化 standard_id → 与 matched_spectrum_id 对齐
            # ======================================================
            raw_id = meta_lower.get("standard_id")
            if raw_id:
                try:
                    standard_id = str(int(float(raw_id)))  # "6812.0" → "6812"
                except:
                    standard_id = str(raw_id).strip()
            else:
                standard_id = None

            # ======================================================
            #  4) pepmass：优先从解析表中取 → 没有则尝试 meta 内字段
            # ======================================================
            pepmass = title_pepmass_map.get(title_str)

            if pepmass is None:
                if "pepmass" in meta_lower:
                    try:
                        pepmass = float(meta_lower["pepmass"].split()[0])
                    except:
                        pepmass = None

            if pepmass is None and "precursor_mz" in meta_lower:
                try:
                    pepmass = float(meta_lower["precursor_mz"])
                except:
                    pepmass = None

            # 最终兜底
            if pepmass is None:
                pepmass = 0.0

            # ======================================================
            #  5) precursor_mz 优先填 pepmass（确保不为 0）
            # ======================================================
            try:
                precursor_val = float(meta_lower.get("precursor_mz") or 0)
            except:
                precursor_val = 0.0

            if precursor_val == 0:
                precursor_val = pepmass

            print(f"Importing title={title_str}, pepmass={pepmass}, precursor={precursor_val}")

            # ======================================================
            #  6) 创建对象
            # ======================================================
            obj = CompoundLibrary(
                spectrum_type='standard',
                ionmode=ionmode,
                title=title_str,
                standard=standard_str,
                database=meta_lower.get('database') or 'standard',
                smiles=meta_lower.get('smiles') or '',
                score=float(meta_lower.get('score') or 0),
                precursor_mz=precursor_val,
                rtinseconds=float(meta_lower.get('rtinseconds') or 0),
                pepmass=pepmass,
                standard_id=standard_id,
                plants=meta_lower.get('plants') or meta_lower.get('PLANTS') or '',
                spectrum_blob=pickle.dumps(spec),
                peaks=[
                    {"mz": float(m), "int": float(i)}
                    for m, i in zip(spec.peaks.mz, spec.peaks.intensities)
                ]
            )

            obj.save()
            count += 1

        self.stdout.write(self.style.SUCCESS(
            f"[✓] Imported {count} / {total} spectra from {mgf_path}"
        ))
