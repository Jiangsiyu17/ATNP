# web/management/commands/import_standards.py
from django.core.management.base import BaseCommand
from matchms.importing import load_from_mgf
from web.models import CompoundLibrary
import pickle
import json

MAX_CHAR_LENGTH = 255


class Command(BaseCommand):
    help = 'Import a standard MGF (positive or negative) into CompoundLibrary'

    def add_arguments(self, parser):
        parser.add_argument('mgf_path', type=str)
        parser.add_argument(
            '--ionmode',
            choices=['positive', 'negative'],
            required=True
        )

    def handle(self, mgf_path, ionmode, **opts):

        # ==========================================================
        #  1) 手动解析 MGF 原文：TITLE / PEPMASS / PRECURSOR_MZ / PLANTS
        # ==========================================================
        title_pepmass_map = {}
        title_plants_map = {}

        with open(mgf_path, 'r', encoding='utf-8') as f:
            block = []
            for line in f:
                line = line.strip()

                if line == "BEGIN IONS":
                    block = []

                elif line == "END IONS":
                    title = None
                    pepmass_val = None
                    precursor_val = None
                    plants_json = None

                    for l in block:
                        if l.startswith("TITLE="):
                            title = l.replace("TITLE=", "").strip()[:MAX_CHAR_LENGTH]

                        elif l.startswith("PEPMASS="):
                            try:
                                pepmass_val = float(l.split("=", 1)[1].split()[0])
                            except Exception:
                                pass

                        elif l.startswith("PRECURSOR_MZ="):
                            try:
                                precursor_val = float(l.split("=", 1)[1].split()[0])
                            except Exception:
                                pass

                        elif l.startswith("PLANTS="):
                            try:
                                plants_json = json.loads(
                                    l.replace("PLANTS=", "", 1)
                                )
                            except Exception as e:
                                self.stderr.write(
                                    f"[WARN] PLANTS JSON parse failed: {e}"
                                )

                    final_pepmass = pepmass_val or precursor_val

                    if title and final_pepmass:
                        title_pepmass_map[title] = final_pepmass

                    if title and plants_json:
                        title_plants_map[title] = plants_json

                else:
                    block.append(line)

        self.stdout.write(
            self.style.SUCCESS(
                f"[✓] Parsed raw MGF blocks: "
                f"{len(title_pepmass_map)} pepmass, "
                f"{len(title_plants_map)} plants"
            )
        )

        # ==========================================================
        #  2) 用 matchms 读取谱图数据（只拿 Spectrum）
        # ==========================================================
        total, count = 0, 0

        for spec in load_from_mgf(mgf_path):
            total += 1

            meta = spec.metadata
            meta_lower = {k.lower(): v for k, v in meta.items()}

            title_str = (
                meta_lower.get('title')
                or meta_lower.get('compound_name')
                or ''
            )[:MAX_CHAR_LENGTH]

            standard_str = (
                meta_lower.get('compound_name')
                or meta_lower.get('title')
                or ''
            )[:MAX_CHAR_LENGTH]

            # ======================================================
            #  standard_id：与 matched_spectrum_id 对齐
            # ======================================================
            raw_id = meta_lower.get("standard_id")
            if raw_id:
                try:
                    standard_id = str(int(float(raw_id)))
                except Exception:
                    standard_id = str(raw_id).strip()
            else:
                standard_id = None

            # ======================================================
            #  pepmass / precursor_mz
            # ======================================================
            pepmass = title_pepmass_map.get(title_str)

            if pepmass is None:
                try:
                    pepmass = float(meta_lower.get("pepmass", 0))
                except Exception:
                    pepmass = 0.0

            try:
                precursor_val = float(meta_lower.get("precursor_mz") or 0)
            except Exception:
                precursor_val = 0.0

            if precursor_val == 0:
                precursor_val = pepmass

            # ======================================================
            #  PLANTS（关键）
            # ======================================================
            plants = title_plants_map.get(title_str)

            self.stdout.write(
                f"Importing: {title_str} | pepmass={pepmass} | plants={'YES' if plants else 'NO'}"
            )

            # ======================================================
            #  创建并保存对象
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

                # ⭐ 关键：JSONField 写 list[dict]
                plants=plants,

                spectrum_blob=pickle.dumps(spec),
                peaks=[
                    {"mz": float(m), "int": float(i)}
                    for m, i in zip(
                        spec.peaks.mz,
                        spec.peaks.intensities
                    )
                ]
            )

            obj.save()
            count += 1

        self.stdout.write(
            self.style.SUCCESS(
                f"[✓] Imported {count} / {total} spectra from {mgf_path}"
            )
        )
