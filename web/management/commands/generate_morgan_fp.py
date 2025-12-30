from django.core.management.base import BaseCommand
from web.models import CompoundLibrary
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from tqdm import tqdm
import warnings
from django.conf import settings

class Command(BaseCommand):
    help = "Recalculate Morgan FP only for RDKit-valid SMILES"
    def handle(self, *args, **kwargs):
        self.stdout.write("=== DEBUG DATABASE INFO ===")
        self.stdout.write(f"DB ENGINE: {settings.DATABASES['default']['ENGINE']}")
        self.stdout.write(f"DB NAME  : {settings.DATABASES['default']['NAME']}")
        self.stdout.write(f"DB HOST  : {settings.DATABASES['default'].get('HOST')}")
        self.stdout.write("===========================")

        self.stdout.write(f"TOTAL COUNT   : {CompoundLibrary.objects.count()}")
        self.stdout.write(
            f"STANDARD COUNT: {CompoundLibrary.objects.filter(spectrum_type__iexact='standard').count()}"
        )
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

        self.stdout.write(f"Recalculating Morgan FP for {total} valid compounds...")

        for c in tqdm(qs.iterator(), total=total):
            # 已有指纹就跳过（可选）
            if c.morgan_fp:
                skipped += 1
                continue

            try:
                mol = Chem.MolFromSmiles(c.smiles)
                if mol is None:
                    continue

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fp = AllChem.GetMorganFingerprintAsBitVect(
                        mol, radius=2, nBits=2048
                    )

                c.morgan_fp = DataStructs.BitVectToBinaryText(fp)
                c.save(update_fields=["morgan_fp"])
                updated += 1

            except Exception:
                continue

        self.stdout.write(self.style.SUCCESS("Done"))
        self.stdout.write(f"✔ Updated: {updated}")
        self.stdout.write(f"↪ Skipped (already had fp): {skipped}")
