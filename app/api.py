from io import BytesIO

from flask import Flask, jsonify, request, render_template_string
from PIL import Image, UnidentifiedImageError

from app.inference import ModelInference


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

inference_engine = ModelInference()


HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Humanli AI Classifier</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 760px;
            margin: 45px auto;
            padding: 20px;
            background: #f4f6f9;
            color: #202938;
        }
        .panel {
            background: white;
            padding: 28px;
            border-radius: 12px;
            box-shadow: 0 3px 16px #00000012;
        }
        h1 { margin-top: 0; }
        input, select, button {
            width: 100%;
            padding: 12px;
            margin: 12px 0;
            box-sizing: border-box;
        }
        button {
            background: #243b70;
            color: white;
            border: 0;
            border-radius: 6px;
            cursor: pointer;
        }
        pre {
            background: #f0f2f5;
            padding: 16px;
            border-radius: 8px;
            white-space: pre-wrap;
            overflow-wrap: anywhere;
        }
        .error { color: #b42318; }
    </style>
</head>
<body>
<div class="panel">
    <h1>Humanli Animal Classifier</h1>
    <p>Upload an image to classify it as Cat, Dog, or Wild.</p>

    <form action="/classify" method="POST" enctype="multipart/form-data">
        <label><b>Select an image</b></label>
        <input type="file" name="file" accept="image/*" required>

        <label><b>Choose inference engine</b></label>
        <select name="model">
            <option value="pytorch">PyTorch Native</option>
            <option value="onnx">ONNX Runtime</option>
            <option value="compare">Compare Both</option>
        </select>

        <button type="submit">Run Prediction</button>
    </form>

    {% if error %}
        <p class="error">{{ error }}</p>
    {% endif %}

    {% if result %}
        <h2>Prediction Results</h2>
        <pre>{{ result | tojson(indent=2) }}</pre>
    {% endif %}
</div>
</body>
</html>
"""


def read_uploaded_image():
    if "file" not in request.files:
        raise ValueError("Upload an image using the 'file' field.")

    uploaded_file = request.files["file"]

    if not uploaded_file.filename:
        raise ValueError("No file selected.")

    image_bytes = uploaded_file.read()

    if not image_bytes:
        raise ValueError("Uploaded file is empty.")

    image = Image.open(BytesIO(image_bytes))
    image.load()
    return image


@app.get("/")
def index():
    return render_template_string(HTML_PAGE)


@app.get("/health")
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": inference_engine.model is not None,
        "onnx_loaded": inference_engine.onnx_session is not None,
        "device": str(inference_engine.device)
    })


@app.get("/model-info")
def model_info():
    return jsonify({
        "architecture": inference_engine.config["architecture"],
        "input_size": inference_engine.config["image_size"],
        "num_classes": inference_engine.config["num_classes"],
        "classes": list(inference_engine.class_mapping.keys()),
        "parameters": inference_engine.config["parameters"],
        "inference_engines": ["pytorch", "onnx", "compare"]
    })


@app.post("/predict")
def predict():
    try:
        image = read_uploaded_image()
        backend = request.form.get("model", "pytorch").lower()

        result = inference_engine.predict(image, backend)

        return jsonify({
            "success": True,
            "model": backend,
            "result": result
        }), 200

    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    except (UnidentifiedImageError, OSError):
        return jsonify({
            "error": "Invalid or unsupported image file."
        }), 400


@app.post("/classify")
def classify():
    try:
        image = read_uploaded_image()
        backend = request.form.get("model", "pytorch").lower()
        result = inference_engine.predict(image, backend)

        return render_template_string(
            HTML_PAGE, result=result, error=None
        )

    except (ValueError, UnidentifiedImageError, OSError) as error:
        return render_template_string(
            HTML_PAGE, result=None, error=str(error)
        ), 400


@app.errorhandler(413)
def file_too_large(error):
    return jsonify({
        "error": "Image exceeds the 10 MB upload limit."
    }), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)