import pytest

from autoaudit_llm.api import create_app
from autoaudit_llm.config import load_config
from autoaudit_llm.inference import FIRGenerator


def test_generator_rule_based_mode():
    gen = FIRGenerator(prefer_model=False)
    assert gen.using_model is False
    complaint = gen.generate("আমার মানিব্যাগ চুরি হয়েছে উত্তরা বিকেল ৫টা")
    assert complaint.offense_type == "চুরি"


def test_generator_rejects_empty():
    gen = FIRGenerator(prefer_model=False)
    with pytest.raises(ValueError):
        gen.generate("   ")


@pytest.fixture
def client():
    app = create_app(config=load_config(), prefer_model=False)
    app.config.update(TESTING=True)
    return app.test_client()


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"


def test_generate_endpoint(client):
    resp = client.post("/generate", json={"raw_statement": "আমার মোবাইল ফোন ছিনতাই মিরপুর ১০ রাত ৯টা"})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["complaint"]["offense_type"] == "ছিনতাই"
    assert "প্রথম তথ্য বিবরণী" in body["document"]


def test_generate_endpoint_validation(client):
    resp = client.post("/generate", json={"raw_statement": ""})
    assert resp.status_code == 400


def test_extract_endpoint(client):
    resp = client.post("/extract", json={"raw_statement": "করিম মিয়া আমাকে হুমকি দিয়েছে বাড্ডা রাত ১১টা"})
    assert resp.status_code == 200
    ext = resp.get_json()["extraction"]
    assert ext["offense_type"] == "হুমকি প্রদান"
