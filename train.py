"""Train SimpleCNN or ResNet18 on data/Training.

A random, stratified validation split is held out from Training (image level:
the dataset has no patient IDs). The best checkpoint on validation accuracy is
saved to models/<model>.pth.

    python train.py --model simplecnn --epochs 12 --augment
    python train.py --model resnet18 --epochs 50 --augment
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from src.dataset import BrainMRIDataset, find_classes
from src.models import build_model
from src.utils import evaluate_model, get_device, set_seed


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data/Training")
    p.add_argument("--model", choices=["simplecnn", "resnet18"], default="simplecnn")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-split", type=float, default=0.15)
    p.add_argument("--augment", action="store_true", help="blur / noise / motion blur on training images")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--out-dir", default="models")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    classes = find_classes(args.data_dir)
    print(f"Classes: {classes} | device: {device}")

    # Two views of the same files: augmented for training, clean for validation
    train_view = BrainMRIDataset(args.data_dir, classes, augment=args.augment)
    val_view = BrainMRIDataset(args.data_dir, classes, augment=False)

    indices = np.arange(len(train_view))
    train_idx, val_idx = train_test_split(
        indices, test_size=args.val_split, stratify=train_view.labels, random_state=args.seed
    )
    train_loader = DataLoader(Subset(train_view, train_idx), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(Subset(val_view, val_idx), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)
    print(f"Train: {len(train_idx)} images | Val: {len(val_idx)} images")

    model = build_model(args.model, num_classes=len(classes)).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, f"{args.model}.pth")
    best_acc = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        train_loss = running_loss / len(train_idx)
        _, val_acc = evaluate_model(model, val_loader, device, num_classes=len(classes))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_acc": float(val_acc)})
        print(f"Epoch {epoch:3d}/{args.epochs} | loss {train_loss:.4f} | val acc {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": args.model, "classes": classes, "state_dict": model.state_dict()}, ckpt_path)

    with open(os.path.join(args.out_dir, f"{args.model}_history.json"), "w") as f:
        json.dump({"args": vars(args), "best_val_acc": float(best_acc), "history": history}, f, indent=2)
    print(f"\nBest val acc: {best_acc:.4f} -> {ckpt_path}")


if __name__ == "__main__":
    main()
