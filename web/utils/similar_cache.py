# web/utils/similar_cache.py
import pickle
import logging

from django.conf import settings


logger = logging.getLogger(__name__)

POS_PATH = settings.BASE_DIR / "model" / "compound_similar_samples_pos.pickle"
NEG_PATH = settings.BASE_DIR / "model" / "compound_similar_samples_neg.pickle"

_cache = {"pos": None, "neg": None}


def get_similar_samples(compound_id, ionmode="positive"):
    key = "pos" if ionmode.startswith("pos") else "neg"

    if _cache[key] is None:
        path = POS_PATH if key == "pos" else NEG_PATH
        try:
            with path.open("rb") as f:
                _cache[key] = pickle.load(f)
        except (OSError, pickle.UnpicklingError, EOFError):
            logger.exception("Unable to load similar-compound cache: %s", path)
            _cache[key] = {}

    return _cache[key].get(compound_id, [])
