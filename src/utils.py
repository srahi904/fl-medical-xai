"""
utils.py
--------
Shared training / evaluation loops and metric helpers used by both the
centralized baseline and every federated client.
"""

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np


def train_one_epoch(model, dataloader, optimizer, device, criterion=None):
    """Runs one local training epoch. Used identically by centralized training
    and by each FL client during its local update step."""
    model.train()
    criterion = criterion or nn.CrossEntropyLoss()
    running_loss, total = 0.0, 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        total += images.size(0)

    return running_loss / max(total, 1)


@torch.no_grad()
def evaluate(model, dataloader, device, criterion=None):
    """Evaluate model, return dict of loss, accuracy, f1, auc."""
    model.eval()
    criterion = criterion or nn.CrossEntropyLoss()

    all_preds, all_labels, all_probs = [], [], []
    running_loss, total = 0.0, 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        running_loss += loss.item() * images.size(0)
        total += images.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    all_probs = np.array(all_probs)
    metrics = {
        "loss": running_loss / max(total, 1),
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
    }

    # AUC only makes sense with >=2 classes present and probability scores
    try:
        if all_probs.shape[1] == 2:
            metrics["auc"] = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            metrics["auc"] = roc_auc_score(all_labels, all_probs, multi_class="ovr")
    except ValueError:
        metrics["auc"] = float("nan")  # happens if a batch/test set has only 1 class

    return metrics


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
