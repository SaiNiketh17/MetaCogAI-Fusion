# main.py

from models import predict_image, predict_text, fuse_predictions
import torch
import random
from torchvision import datasets, transforms
from datasets import load_dataset
from utils.logger import log_prediction

# ------------------- Load Random CIFAR-10 Image -------------------
def get_random_cifar10_image():
    transform = transforms.Compose([transforms.ToTensor()])
    cifar10 = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    idx = random.randint(0, len(cifar10) - 1)
    image, label = cifar10[idx]
    return image, label

# ------------------- Load Random IMDb Review -------------------
def get_random_imdb_review():
    dataset = load_dataset("imdb", split="test")
    idx = random.randint(0, len(dataset) - 1)
    return dataset[idx]["text"]

# ------------------- Test Image Prediction -------------------
def test_image_prediction():
    image_tensor, _ = get_random_cifar10_image()
    label, confidences = predict_image(image_tensor)

    print("📷 Image Prediction:", label)
    print("🔢 Top 3 Confidences:", [round(c, 4) for c in confidences])

    log_prediction(
        modality="image",
        input_summary="CIFAR-10 image",
        prediction=label,
        confidence=confidences[0]
    )
    return label, confidences[0]

# ------------------- Test Text Prediction -------------------
def test_text_prediction():
    text = get_random_imdb_review()
    label, confidence = predict_text(text)

    print("📝 Text Sentiment Prediction:", label)
    print("📈 Sentiment Confidence:", round(confidence, 4))

    log_prediction(
        modality="text",
        input_summary=text[:50],
        prediction=label,
        confidence=confidence
    )
    return label, confidence

# ------------------- Fusion Prediction Test -------------------
def test_fusion():
    print("\n🔀 --- Fusion Module Test ---\n")
    image_label, image_conf = test_image_prediction()
    print("\n---\n")
    text_label, text_conf = test_text_prediction()
    print("\n---\n")

    final_label, final_conf, rationale = fuse_predictions(
        (image_label, image_conf),
        (text_label, text_conf)
    )

    print("🧠 Final Decision:", final_label)
    print("💡 Final Confidence:", round(final_conf, 4))
    print("📚 Rationale:", rationale)

    log_prediction(
        modality="fusion",
        input_summary=f"Image={image_label} ({image_conf:.3f}), Text={text_label} ({text_conf:.3f})",
        prediction=final_label,
        confidence=final_conf
    )

# ------------------- Main -------------------
if __name__ == "__main__":
    print("\n🌟 Individual Predictions:\n")
    test_image_prediction()
    print("\n---\n")
    test_text_prediction()
    print("\n==============================\n")
    test_fusion()
