"""Runtime data paths shared by standalone and integrated deployments."""

import os
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(
    os.getenv("ATNP_QA_DATA_DIR", str(PROJECT_DIR / "data"))
).expanduser()

KG_DB_PATH = Path(
    os.getenv("ATNP_KG_DB", str(DATA_DIR / "knowledge_base_full.duckdb"))
).expanduser()

PASSAGE_INDEX_PATH = Path(
    os.getenv("ATNP_PASSAGE_INDEX", str(DATA_DIR / "passages.sqlite"))
).expanduser()
