from io import BytesIO

import pytest
from PIL import Image

from app.api import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json["status"] == "healthy"
    assert response.json["model_loaded"] is True


def test_model_info_endpoint(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    assert response.json["architecture"] == "CustomCNN"
    assert response.json["num_classes"] == 3
    assert set(response.json["classes"]) == {"cat", "dog", "wild"}


def test_prediction_endpoint(client):
    image = Image.new("RGB", (224, 224), color=(120, 100, 80))
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)

    response = client.post(
        "/predict",
        data={"file": (buffer, "test.jpg")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    result = response.json["result"]
    assert result["predicted_class"] in {"cat", "dog", "wild"}
    assert 0 <= result["confidence"] <= 1
    assert set(result["probabilities"]) == {"cat", "dog", "wild"}
    assert sum(result["probabilities"].values()) == pytest.approx(
        1.0, abs=0.001
    )


def test_missing_file(client):
    response = client.post("/predict")
    assert response.status_code == 400


def test_invalid_image(client):
    response = client.post(
        "/predict",
        data={"file": (BytesIO(b"not an image"), "invalid.jpg")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400