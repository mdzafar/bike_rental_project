from fastapi.testclient import TestClient
from app.main import app
from pytest import approx

client = TestClient(app)

def test_index_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/html; charset=utf-8"
    
def test_health_main():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["api_version"] == "0.0.1"
    assert response.json()["model_version"] == "0.0.1"
    
def test_predit_main():
    payload = {
        "inputs" : [
            {
                "dteday": "2012-05-13",
                "season": "summer",
                "hr": "12pm",
                "holiday": "No",
                "weekday": "Sun",
                "workingday": "No",
                "weathersit": "Clear",
                "temp": 22.08,
                "atemp": 24.99,
                "hum": 56.99,
                "windspeed": 15.00,
                "casual": 189,
                "registered": 342,
            }
        ]
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["errors"] is None
    assert isinstance(data["predictions"], float)
    assert data["predictions"] == approx(599.06, rel=1e-2)
