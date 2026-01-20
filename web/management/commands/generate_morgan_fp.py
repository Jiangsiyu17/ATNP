from django.core.management.base import BaseCommand
from web.models import CompoundLibrary
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from tqdm import tqdm
from django.conf import settings
import warnings

from web.utils.compound_aggregate import normalize_mol  # ⭐ 新增

class Command(BaseCommand):
    help = "Recalculate Morgan FP for standard compounds (WITH normalization)"

    def handle(self, *args, **kwargs):
        self.stdout.write("=== DEBUG DATABASE INFO ===")
        self.stdout.write(f"DB ENGINE: {settings.DATABASES['default']['ENGINE']}")
        self.stdout.write(f"DB NAME  : {settings.DATABASES['default']['NAME']}")
        self.stdout.write("===========================")

        qs = CompoundLibrary.objects.filter(
            spectrum_type__iexact="standard"
        ).exclude(
            smiles__isnull=True
        ).exclude(
            smiles=""
        )

        total = qs.count()
        updated = 0
        skipped = 0

        self.stdout.write(f"Recalculating Morgan FP for {total} compounds (normalized)...")

        for c in tqdm(qs.iterator(), total=total):
            try:
                mol = Chem.MolFromSmiles(c.smiles)
                if mol is None:
                    c.morgan_fp = None
                    c.save(update_fields=["morgan_fp"])
                    skipped += 1
                    continue

                mol = normalize_mol(mol)  # ⭐ 关键

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fp = AllChem.GetMorganFingerprintAsBitVect(
                        mol, radius=2, nBits=2048
                    )

                c.morgan_fp = DataStructs.BitVectToBinaryText(fp)
                c.save(update_fields=["morgan_fp"])
                updated += 1

            except Exception:
                c.morgan_fp = None
                c.save(update_fields=["morgan_fp"])
                skipped += 1

        self.stdout.write(self.style.SUCCESS("Done"))
        self.stdout.write(f"✔ Updated: {updated}")
        self.stdout.write(f"↪ Skipped: {skipped}")
