"""
train_centralized.py
---------------------
Baseline: trains the SAME model architecture on ALL data pooled together in
one place (the traditional, non-private way). We compare this against the
federated result to show the privacy-vs-accuracy trade-off in the final report:

    "Federated model achieves X% accuracy vs Y% for centralized training,
     while never sharing raw patient data."

Run:
    python src/train_centralized.py --data_dir data/train --test_data_dir data/test
"""

import argparse
import os
import torch

from model import build_model
from dataset import load_full_dataset, get_transforms, get_test_dataloader
from utils import train_one_epoch, evaluate, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/train")
    parser.add_argument("--test_data_dir", type=str, default="data/test")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_classes", type=int, default=2)
    args = parser.parse_args()

    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_set = load_full_dataset(args.data_dir, train=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader, classes = get_test_dataloader(args.test_data_dir, batch_size=args.batch_size)

    model = build_model(num_classes=args.num_classes, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    os.makedirs("saved_models", exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, device)
        metrics = evaluate(model, test_loader, device)
        print(f"Epoch {epoch}/{args.epochs} | train_loss={loss:.4f} | "
              f"test_acc={metrics['accuracy']:.4f} | test_f1={metrics['f1']:.4f} | "
              f"test_auc={metrics['auc']:.4f}")

        if metrics["accuracy"] > best_acc:
            best_acc = metrics["accuracy"]
            torch.save(model.state_dict(), "saved_models/centralized_best.pth")

    print(f"\nBest centralized test accuracy: {best_acc:.4f}")
    print("Saved best model to saved_models/centralized_best.pth")
    print("\nUse this number as the baseline to compare against your federated results (src/server.py).")


if __name__ == "__main__":
    main()
