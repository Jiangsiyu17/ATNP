from pathlib import Path

from django.conf import settings


MAX_QUERY_LENGTH = 120


class LiteratureDatabaseError(RuntimeError):
    """Raised when the curated literature database cannot be queried."""


def _connect():
    try:
        import duckdb
    except ImportError as exc:
        raise LiteratureDatabaseError(
            "The DuckDB Python package is not installed."
        ) from exc

    database_path = Path(settings.BASE_DIR) / "crawler" / "knowledge_base_full.duckdb"
    if not database_path.is_file():
        raise LiteratureDatabaseError("The literature database file was not found.")

    try:
        return duckdb.connect(str(database_path), read_only=True)
    except Exception as exc:
        raise LiteratureDatabaseError(
            "The literature database could not be opened."
        ) from exc


def _normalise_query(query):
    return " ".join((query or "").strip().split())[:MAX_QUERY_LENGTH]


def _like_pattern(value):
    value = value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{value}%"
