from django.core.management.base import BaseCommand
from django.conf import settings
from web.models import CompoundLibrary
from web.utils.identify import find_most_similar_spectrum
import pickle
from tqdm import tqdm

OUT_POS = settings.BASE_DIR / "model" / "compound_similar_samples_pos.pickle"
OUT_NEG = settings.BASE_DIR / "model" / "compound_similar_samples_neg.pickle"

class Command(BaseCommand):
    help = "Precompute spec2vec similar plant spectra for all compounds (score > 0.6)"

    def handle(self, *args, **kwargs):
        results_pos = {}
        results_neg = {}

        results_pos = {}
        results_neg = {}

        base_qs = CompoundLibrary.objects.filter(
            spectrum_type__iexact="standard"
        ).exclude(
            peaks=None
        )

        total = base_qs.count()

        qs = base_qs.iterator(
            chunk_size=500
        )

        for compound in tqdm(
            qs,
            total=total,
            desc="Precomputing"
        ):

            spectrum = compound.get_spectrum()

            if spectrum is None:
                continue

            ionmode = (
                compound.ionmode or "positive"
            ).lower()

            res = find_most_similar_spectrum(
                spectrum,
                ionmode=ionmode
            )

            if ionmode.startswith("pos"):
                results_pos[compound.id] = res
            else:
                results_neg[compound.id] = res

        with open(OUT_POS, "wb") as f:
            pickle.dump(
                results_pos,
                f
            )

        with open(OUT_NEG, "wb") as f:
            pickle.dump(
                results_neg,
                f
            )

        self.stdout.write(
            self.style.SUCCESS(
                f"✅ finished: "
                f"pos={len(results_pos)} "
                f"neg={len(results_neg)}"
            )
        )
