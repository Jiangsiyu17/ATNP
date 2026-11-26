from app.api import app


def test_integrated_service_exposes_required_api_routes_only():
    paths = {route.path for route in app.routes}

    assert "/expert-qa/api/health" in paths
    assert "/expert-qa/api/models" in paths
    assert "/expert-qa/api/chat/stream" in paths
    assert "/expert-qa" not in paths


def test_runtime_database_files_are_not_bundled():
    service_root = __import__("pathlib").Path(__file__).resolve().parents[1]

    assert not list(service_root.rglob("*.duckdb"))
    assert not list(service_root.rglob("*.sqlite"))
