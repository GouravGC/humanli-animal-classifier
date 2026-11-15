
# Humanli Animal Image Classification API

Production-oriented REST API for classifying animal face images into three categories:

- Cat
- Dog
- Wild

## Model

- Architecture: Custom CNN
- Framework: PyTorch
- Input: RGB images, 224 × 224
- Parameters: 1,206,499
- Test accuracy: 97%
- Test macro F1: 97%

## Features

- Flask REST API
- Image preprocessing and validation
- CPU-based inference
- Model loaded once at application startup
- Automated API tests
- Docker deployment
- GitHub Actions CI

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API information |
| GET | `/health` | Health check |
| GET | `/model-info` | Model metadata |
| POST | `/predict` | Image classification |

## Run locally

```bash
pip install -r requirements.txt
python -m app.api
```

## Prediction

```bash
curl -X POST \
  -F "file=@test_image.jpg" \
  http://localhost:5000/predict
```

## Run tests

```bash
python -m pytest tests -v
```

## Docker

```bash
docker build -t humanli-classifier .
docker run -p 5000:5000 humanli-classifier
```

## Model evaluation

Held-out test set:
- Accuracy: 97%
- Macro precision: 97.10%
- Macro recall: 97%
- Macro F1: 97%

Training and evaluation artifacts are retained in the project backup.
