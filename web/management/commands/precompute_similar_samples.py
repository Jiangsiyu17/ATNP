from django.core.management.base import BaseCommand
from web.models import CompoundLibrary
from web.utils.identify import find_most_similar_spectrum
import pickle
from tqdm import tqdm

OUT_POS = "/data2/jiangsiyu/ATNP_Database/model/compound_similar_samples_pos.pickle"
OUT_NEG = "/data2/jiangsiyu/ATNP_Database/model/compound_similar_samples_neg.pickle"

class Command(BaseCommand):
    help = "Precompute spec2vec similar plant spectra for all compounds (score > 0.6)"

    def handle(self, *args, **kwargs):
        results_pos = {}
        results_neg = {}

        qs = CompoundLibrary.objects.exclude(peaks=None)

        for compound in tqdm(qs, desc="Precomputing"):
            spectrum = compound.get_spectrum()
            if spectrum is None:
                continue

            ionmode = (compound.ionmode or "positive").lower()

            res = find_most_similar_spectrum(
                spectrum,
                ionmode=ionmode
            )

            if ionmode.startswith("pos"):
                results_pos[compound.id] = res
            else:
                results_neg[compound.id] = res

        with open(OUT_POS, "wb") as f:
            pickle.dump(results_pos, f)

        with open(OUT_NEG, "wb") as f:
            pickle.dump(results_neg, f)

        self.stdout.write(self.style.SUCCESS(
            f"✅ Precompute finished: "
            f"pos={len(results_pos)}, neg={len(results_neg)}"
        ))
