# multimodal_model.py

# ================================
# MetaCogAI - Predictive Module
# Phase 1 - Task 2
# ================================

import torch
from PIL import Image
from typing import Tuple
from torchvision import transforms, models
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from utils.logger import log_prediction
from pathlib import Path

# -------------------------------
# Load Trained ResNet-18 Model
# -------------------------------
def load_trained_resnet(model_path: str) -> torch.nn.Module:
    model = models.resnet18(pretrained=False, num_classes=10)
    model.fc = torch.nn.Linear(model.fc.in_features, 10)  # CIFAR-10
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model


# Set up model paths
image_model_path = Path(__file__).resolve().parent / "image_model.pth"
text_model_dir = Path(__file__).resolve().parent / "text_model"

# Load models
resnet_model = load_trained_resnet(str(image_model_path))

tokenizer = DistilBertTokenizerFast.from_pretrained(str(text_model_dir))
bert_model = DistilBertForSequenceClassification.from_pretrained(str(text_model_dir))
bert_model.eval()




# CIFAR-10 Class Labels
cifar10_classes = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


# Predict from Image
# -------------------
def predict_image(image: Image.Image) -> Tuple[str, float]:
    if isinstance(image, torch.Tensor):
        if len(image.shape) == 3:
            image_tensor = image.unsqueeze(0)  # [1, 3, 32, 32]
        elif len(image.shape) == 4:
            image_tensor = image
        else:
            raise ValueError("Invalid image tensor shape.")
    else:
        transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Optional if using CIFAR-10 size
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


        image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = resnet_model(image_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top_probs, top_indices = torch.topk(probabilities, 3)

    predicted_class = top_indices[0].item()
    confidence_score = top_probs[0].item()

    # Safeguard against index out of range
    if predicted_class >= len(cifar10_classes):
        label = f"class_{predicted_class}"
    else:
        label = cifar10_classes[predicted_class]

    # Log the prediction
    log_prediction(
        modality="image",
        input_summary="CIFAR-10 image",
        prediction=label,
        confidence=confidence_score
    )

    return label, top_probs.tolist()



# Predict from Text
# ------------------
def predict_text(text: str) -> Tuple[str, float]:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = bert_model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

    predicted_class = torch.argmax(probabilities, dim=1).item()
    confidence_score = probabilities[0][predicted_class].item()
    sentiment = "positive" if predicted_class == 1 else "negative"

    # Log the prediction
    log_prediction(
        modality="text",
        input_summary=text[:50],
        prediction=sentiment,
        confidence=confidence_score
    )

    return sentiment, confidence_score


# Simulated Fusion (for Phase 2)
# -------------------------------
def fuse_predictions(image_label: str, text_label: str) -> str:
    """
    Dummy fusion logic (for Phase 2).
    """
    if image_label in ["cat", "dog"] and text_label == "positive":
        return "Happy Pet Scenario"
    else:
        return "Generic Case"
