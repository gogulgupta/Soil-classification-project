"""
Model Training Pipeline for Soil Classification
Author: Gogul Gupta (Team Shree Yantra Dynamics)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import build_soil_classifier, CLASS_NAMES
from src.dataset import SoilDataset, get_transforms


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels, _ in tqdm(dataloader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_f1 = f1_score(all_labels, all_preds, average="macro")
    return epoch_loss, epoch_f1


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = running_loss / len(dataloader.dataset)
    val_f1_macro = f1_score(all_labels, all_preds, average="macro")
    per_class_f1 = f1_score(all_labels, all_preds, average=None)
    min_class_f1 = float(np.min(per_class_f1))

    return val_loss, val_f1_macro, min_class_f1, all_labels, all_preds


def train_pipeline(
    train_dir: str,
    train_csv: str,
    output_model_path: str = "models/best_model.pth",
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-4,
    k_folds: int = 5
):
    """
    Executes the training and validation workflow with Stratified K-Fold CV.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on device: {device}")

    df = pd.read_csv(train_csv)
    # Ensure numerical labels
    if df["label"].dtype == object or isinstance(df["label"].iloc[0], str):
        class_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}
        df["label"] = df["label"].map(class_to_idx)

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    os.makedirs(os.path.dirname(output_model_path) or ".", exist_ok=True)

    best_global_min_f1 = 0.0

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["label"])):
        print(f"\n{'='*20} Fold {fold + 1} / {k_folds} {'='*20}")
        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_val = df.iloc[val_idx].reset_index(drop=True)

        train_ds = SoilDataset(train_dir, df_train, transform=get_transforms(is_train=True))
        val_ds = SoilDataset(train_dir, df_val, transform=get_transforms(is_train=False))

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

        model = build_soil_classifier(num_classes=len(CLASS_NAMES), pretrained=True).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_fold_f1 = 0.0

        for epoch in range(epochs):
            tr_loss, tr_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_f1, min_f1, _, _ = evaluate(model, val_loader, criterion, device)
            scheduler.step()

            print(f"Epoch [{epoch+1:02d}/{epochs:02d}] "
                  f"Train Loss: {tr_loss:.4f} | Train F1: {tr_f1:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Macro F1: {val_f1:.4f} | Min Class F1: {min_f1:.4f}")

            if min_f1 > best_global_min_f1:
                best_global_min_f1 = min_f1
                torch.save(model.state_dict(), output_model_path)
                print(f"  ✨ New Best Global Model Saved to {output_model_path} (Min F1: {min_f1:.4f})")

    print(f"\n🏆 Training Finished! Best Minimum Class F1: {best_global_min_f1:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train Soil Classification Model")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to training images directory")
    parser.add_argument("--csv", type=str, required=True, help="Path to train labels CSV")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--output", type=str, default="models/best_model.pth", help="Checkpoint save path")

    args = parser.parse_args()
    train_pipeline(
        train_dir=args.data_dir,
        train_csv=args.csv,
        output_model_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )


if __name__ == "__main__":
    main()
