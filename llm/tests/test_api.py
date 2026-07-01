import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.app import create_app  # noqa: E402
from src.schema import FIRRecord, OffenseType, Person  # noqa: E402


def _valid_raw_output(input_text: str) -> str:
    record = FIRRecord(
        offense_type=OffenseType.THEFT,
        complainant=Person(name="করিম উদ্দিন"),
        incident_location="উত্তরা",
        narrative_bn="অভিযোগকারীর মানিব্যাগ চুরি হয়েছে।",
        raw_input_text=input_text,
    )
    return record.model_dump_json()


@pytest.fixture
def client():
    app = create_app(generate_fn=_valid_raw_output)
    app.testing = True
    return app.test_client()


@pytest.fixture
def client_no_model():
    app = create_app(generate_fn=None)
    app.testing = True
    return app.test_client()


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"
    assert resp.get_json()["model_loaded"] is True


def test_generate_fir_success(client):
    resp = client.post(
        "/api/v1/generate-fir",
        data=json.dumps({"text": "আমার মানিব্যাগ চুরি হয়েছে উত্তরায় বিকাল ৫টায়"}),
        content_type="application/json",
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["ok"] is True
    assert body["fir_record"]["offense_type"] == "চুরি"
    assert "FIR" in body["document_text"] or "অভিযোগ" in body["document_text"]


def test_generate_fir_missing_text_returns_400(client):
    resp = client.post("/api/v1/generate-fir", data=json.dumps({}), content_type="application/json")
    assert resp.status_code == 400


def test_generate_fir_without_loaded_model_returns_503(client_no_model):
    resp = client_no_model.post(
        "/api/v1/generate-fir",
        data=json.dumps({"text": "কিছু ইনপুট"}),
        content_type="application/json",
    )
    assert resp.status_code == 503


def test_generate_fir_bad_model_output_returns_422():
    app = create_app(generate_fn=lambda text: "এটি কোনো JSON নয়")
    app.testing = True
    client = app.test_client()
    resp = client.post(
        "/api/v1/generate-fir",
        data=json.dumps({"text": "কিছু ইনপুট"}),
        content_type="application/json",
    )
    assert resp.status_code == 422
    assert resp.get_json()["ok"] is False
