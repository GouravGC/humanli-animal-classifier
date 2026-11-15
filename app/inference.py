import json
import time
from pathlib import Path

import numpy as np
import torch
import onnxruntime as ort

from model.custom_cnn import CustomCNN
from app.preprocessing import preprocess_image


ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = ROOT_DIR / "artifacts"


class ModelInference:
    def __init__(self):
        self.device = torch.device("cpu")

        with open(ARTIFACT_DIR / "model_config.json", encoding="utf-8") as f:
            self.config = json.load(f)

        with open(ARTIFACT_DIR / "class_mapping.json", encoding="utf-8") as f:
            self.class_mapping = json.load(f)

        self.index_to_class = {
            index: label for label, index in self.class_mapping.items()
        }

        # Load native PyTorch model
        self.model = CustomCNN(num_classes=self.config["num_classes"])
        weights_path = ARTIFACT_DIR / "custom_cnn_state_dict.pth"

        state_dict = torch.load(
            weights_path,
            map_location=self.device,
            weights_only=True
        )

        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()

        # Load ONNX model (external data file must be beside the ONNX file)
        onnx_path = ARTIFACT_DIR / "custom_cnn.onnx"
        self.onnx_session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"]
        )
        self.onnx_input_name = self.onnx_session.get_inputs()[0].name

    def _format_result(self, logits, elapsed_ms):
        probabilities = torch.softmax(
            torch.as_tensor(logits), dim=1
        )[0]

        predicted_index = int(torch.argmax(probabilities).item())

        return {
            "predicted_class": self.index_to_class[predicted_index],
            "confidence": round(
                float(probabilities[predicted_index].item()), 6
            ),
            "probabilities": {
                self.index_to_class[i]: round(
                    float(probabilities[i].item()), 6
                )
                for i in range(len(self.index_to_class))
            },
            "inference_time_ms": round(elapsed_ms, 3)
        }

    @torch.inference_mode()
    def predict_pytorch(self, image):
        start = time.perf_counter()

        tensor = preprocess_image(image).to(self.device)
        logits = self.model(tensor).cpu().numpy()

        elapsed_ms = (time.perf_counter() - start) * 1000
        return self._format_result(logits, elapsed_ms)

    def predict_onnx(self, image):
        start = time.perf_counter()

        tensor = preprocess_image(image)
        input_array = tensor.numpy().astype(np.float32)

        logits = self.onnx_session.run(
            None,
            {self.onnx_input_name: input_array}
        )[0]

        elapsed_ms = (time.perf_counter() - start) * 1000
        return self._format_result(logits, elapsed_ms)

    def predict(self, image, backend="pytorch"):
        if backend == "pytorch":
            return self.predict_pytorch(image)

        if backend == "onnx":
            return self.predict_onnx(image)

        if backend == "compare":
            pytorch_result = self.predict_pytorch(image)
            onnx_result = self.predict_onnx(image)

            return {
                "pytorch": pytorch_result,
                "onnx": onnx_result,
                "predictions_match": (
                    pytorch_result["predicted_class"]
                    == onnx_result["predicted_class"]
                )
            }

        raise ValueError("Unsupported model selection.")