## 化合物所在文献部分加载pkl
import pickle

from django.conf import settings


INDEX_PATH = settings.BASE_DIR / "pubmed_index.pkl"

try:
    with INDEX_PATH.open("rb") as f:
        PUBMED_INDEX = pickle.load(f)
except (OSError, pickle.UnpicklingError, EOFError):
    PUBMED_INDEX = {}
