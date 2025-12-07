from django.core.management.base import BaseCommand
from web.models import CompoundLibrary
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import warnings

class Command(BaseCommand):
    help = "Generate correct Morgan fingerprints (binary RDKit format)"

    def handle(self, *args, **kwargs):
        qs = CompoundLibrary.objects.all()
        total = qs.count()

        self.stdout.write(f"Generating Morgan fingerprints for {total} compounds...")

        for i, c in enumerate(qs.iterator(), start=1):
            if not c.smiles:
                continue

            try:
                mol = Chem.MolFromSmiles(c.smiles)
                if mol is None:
                    continue

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)

                # 🔥用 rdkit 官方二进制格式 保存指纹（重点）
                c.morgan_fp = DataStructs.BitVectToBinaryText(fp)

                c.save(update_fields=["morgan_fp"])

            except Exception:
                continue

            if i % 1000 == 0:
                self.stdout.write(f"...processed {i}/{total}")

        self.stdout.write(self.style.SUCCESS("Done."))
