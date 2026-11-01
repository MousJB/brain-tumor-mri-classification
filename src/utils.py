import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_confusion_matrix(cm, classes, title="Confusion Matrix", save_path=None):
    """Shows the figure, or saves it to `save_path` if given."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
        plt.close()
    else:
        plt.show()


def evaluate_model(model, dataloader, device, num_classes=None):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    labels = list(range(num_classes)) if num_classes else None
    cm = confusion_matrix(all_labels, all_preds, labels=labels)
    accuracy = np.trace(cm) / np.sum(cm)
    return cm, accuracy


def test_on_dataset(model, loader, name, device, classes, save_path=None):
    cm, acc = evaluate_model(model, loader, device, num_classes=len(classes))
    print(f"\nDataset: {name} - Accuracy: {acc:.4f}")
    plot_confusion_matrix(cm, classes, title=f"Confusion Matrix - {name}", save_path=save_path)
    return acc
