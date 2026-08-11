"""
gradcam.py
----------
Explainable AI (XAI) layer. Given a trained model + an input image, produces
a Grad-CAM heatmap showing WHICH regions of the image most influenced the
model's prediction -- critical for medical AI, since a doctor needs to know
*why* the model flagged something, not just the label.

Uses the `grad-cam` (pytorch-grad-cam) library, hooked into the last
convolutional block of our ResNet18 backbone (`model.features`).
"""

import numpy as np
import torch
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from dataset import get_transforms, NORMALIZE_MEAN, NORMALIZE_STD, IMG_SIZE


def denormalize(tensor_img: torch.Tensor) -> np.ndarray:
    """Convert a normalized tensor image back to a displayable [0,1] RGB numpy array."""
    img = tensor_img.clone().detach().cpu().numpy().transpose(1, 2, 0)
    mean = np.array(NORMALIZE_MEAN)
    std = np.array(NORMALIZE_STD)
    img = std * img + mean
    return np.clip(img, 0, 1)


def generate_gradcam(model, image_tensor: torch.Tensor, target_class: int = None, device: str = "cpu"):
    """
    Args:
        model: trained MedicalCNN (see model.py)
        image_tensor: single image, shape [3, H, W], already normalized
        target_class: which class to explain. If None, uses the model's
                       own top prediction.
    Returns:
        overlay (H, W, 3) uint8 image with heatmap blended on original,
        predicted_class (int), confidence (float)
    """
    model.eval()
    input_tensor = image_tensor.unsqueeze(0).to(device)

    # last conv block of our backbone -- see model.py: self.features
    target_layers = [model.features[-1]]

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_class = int(torch.argmax(probs, dim=1).item())
        confidence = float(probs[0, pred_class].item())

    explain_class = target_class if target_class is not None else pred_class
    targets = [ClassifierOutputTarget(explain_class)]

    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]  # [H, W]

    rgb_img = denormalize(image_tensor)
    overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    return overlay, pred_class, confidence


def explain_image_file(model, image_path: str, class_names: list, device: str = "cpu"):
    """Convenience wrapper: load an image file from disk, run Grad-CAM, and
    return everything the frontend/backend needs to display a result."""
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    transform = get_transforms(train=False)
    tensor = transform(img)

    overlay, pred_class, confidence = generate_gradcam(model, tensor, device=device)

    return {
        "predicted_class": class_names[pred_class],
        "confidence": round(confidence * 100, 2),
        "heatmap": overlay,  # numpy array, ready to save/encode as PNG
    }


def save_overlay(overlay: np.ndarray, out_path: str):
    """Save a Grad-CAM overlay (RGB numpy array) to disk as PNG."""
    bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, bgr)
