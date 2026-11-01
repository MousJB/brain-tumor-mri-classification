"""Evaluate a checkpoint on one or more test folders.

Each folder must contain one sub-folder per class (same names as in Training).
Accuracies go to results/<model>_metrics.json, confusion matrices to
results/<model>_<folder>.png.

    python evaluate.py --weights models/simplecnn.pth \
        --test-dirs data/Testing data/Blurred data/Noisy data/Motion
"""
import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from src.dataset import BrainMRIDataset
from src.models import build_model
from src.utils import get_device, test_on_dataset


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True)
    p.add_argument("--test-dirs", nargs="+", default=["data/Testing"])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--out-dir", default="results")
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device()

    ckpt = torch.load(args.weights, map_location=device)
    classes = ckpt["classes"]
    model = build_model(ckpt["model"], num_classes=len(classes), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)

    os.makedirs(args.out_dir, exist_ok=True)
    metrics = {}
    for test_dir in args.test_dirs:
        name = os.path.basename(os.path.normpath(test_dir))
        dataset = BrainMRIDataset(test_dir, classes, augment=False)
        if len(dataset) == 0:
            print(f"Skipping {test_dir}: no images found")
            continue
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        save_path = os.path.join(args.out_dir, f"{ckpt['model']}_{name}.png")
        acc = test_on_dataset(model, loader, name, device, classes, save_path=save_path)
        metrics[name] = {"accuracy": float(acc), "n_images": len(dataset)}

    with open(os.path.join(args.out_dir, f"{ckpt['model']}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
