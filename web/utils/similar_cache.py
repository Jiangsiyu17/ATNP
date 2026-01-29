# web/utils/similar_cache.py
import pickle

POS_PATH = "/data2/jiangsiyu/ATNP_Database/model/compound_similar_samples_pos.pickle"
NEG_PATH = "/data2/jiangsiyu/ATNP_Database/model/compound_similar_samples_neg.pickle"

_cache = {"pos": None, "neg": None}

def get_similar_samples(compound_id, ionmode="positive"):
    key = "pos" if ionmode.startswith("pos") else "neg"

    if _cache[key] is None:
        with open(POS_PATH if key == "pos" else NEG_PATH, "rb") as f:
            _cache[key] = pickle.load(f)

    return _cache[key].get(compound_id, [])
