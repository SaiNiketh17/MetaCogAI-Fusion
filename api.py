# app/api.py

from fastapi import APIRouter, File, UploadFile, Form
from pydantic import BaseModel
from typing import Tuple
from PIL import Image
import torch
from torchvision import transforms, models
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from utils.logger import log_prediction
from models.fusion import fuse_predictions
import io

router = APIRouter()

# ------------------- Load Models -------------------
# Load image model
image_model = models.resnet18(num_classes=10)
image_model.load_state_dict(torch.load("models/image_model.pth", map_location=torch.device("cpu")))
image_model.eval()

# Load text model + tokenizer
text_model = DistilBertForSequenceClassification.from_pretrained("models/text_model")
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
text_model.eval()

# CIFAR-10 labels
cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

# ------------------- Utils -------------------
def predict_image(image: Image.Image) -> Tuple[str, float]:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = image_model(img_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)
        top_probs, top_idxs = torch.topk(probs, 3)

    label_idx = top_idxs[0].item()
    label = cifar10_classes[label_idx] if label_idx < len(cifar10_classes) else f"class_{label_idx}"
    confidence = top_probs[0].item()

    log_prediction("image", "API Image", label, confidence)
    return label, confidence

def predict_text(text: str) -> Tuple[str, float]:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = text_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    sentiment = "positive" if pred == 1 else "negative"
    log_prediction("text", text[:50], sentiment, confidence)
    return sentiment, confidence

# ------------------- Routes -------------------

@router.post("/predict/image")
def predict_image_endpoint(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(file.file.read())).convert("RGB")
    label, confidence = predict_image(image)
    return {"label": label, "confidence": round(confidence, 4)}

class TextInput(BaseModel):
    text: str

@router.post("/predict/text")
def predict_text_endpoint(payload: TextInput):
    label, confidence = predict_text(payload.text)
    return {"sentiment": label, "confidence": round(confidence, 4)}

@router.post("/predict/fusion")
def predict_fusion_endpoint(
    file: UploadFile = File(...),
    text: str = Form(...)
):
    image = Image.open(io.BytesIO(file.file.read())).convert("RGB")
    image_label, image_conf = predict_image(image)
    text_label, text_conf = predict_text(text)

    final_label, final_conf, rationale = fuse_predictions((image_label, image_conf), (text_label, text_conf))

    return {
        "final_decision": final_label,
        "confidence": round(final_conf, 4),
        "rationale": rationale
    }
