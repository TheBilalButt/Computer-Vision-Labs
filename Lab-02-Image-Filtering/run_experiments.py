#!/usr/bin/env python3
"""
Lab Task 02: Effect of Image Filtering on Skin-Lesion Classification
Dataset: HAM10000 Skin-Lesion Dataset (ISIC 7-class subset)
Models: ResNet50, ResNet101, VGG16 (Top-3 from Lab 01)
Filters: No Filter, Average/Mean, Gaussian, Median, Sharpening, Sobel Edge
"""

import os, sys, json, time, warnings
from pathlib import Path
from collections import defaultdict
import numpy as np
import cv2
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ----------------- CONFIGURATION -----------------
SEED = 42
np.random.seed(SEED)
IMG_SIZE = 224
MAX_PER_CLASS = 35       # ~245 total images: 196 train, 49 test for rapid, reliable execution
DEVICE = "cpu"

HAM_CLASSES = [
    'actinic keratosis',
    'basal cell carcinoma',
    'dermatofibroma',
    'melanoma',
    'nevus',
    'pigmented benign keratosis',
    'vascular lesion'
]
NUM_CLASSES = len(HAM_CLASSES)

BASE_MODELS = ["ResNet50", "ResNet101", "VGG16"]
FILTER_NAMES = ["No Filter", "Average", "Gaussian", "Median", "Sharpening", "Sobel"]

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, balanced_accuracy_score, roc_auc_score,
                             confusion_matrix)

# ----------------- IMAGE FILTERS -----------------
def apply_filter(img_np, filter_type):
    """
    Apply spatial domain image processing filters:
    - No Filter: Original image
    - Average: 5x5 Mean/Box filter
    - Gaussian: 5x5 Gaussian blur (sigma=1.0)
    - Median: 5x5 Median filter
    - Sharpening: 3x3 unsharp masking kernel
    - Sobel: Sobel gradient magnitude normalized to [0, 255] across 3 channels
    """
    if filter_type == "No Filter":
        return img_np
    elif filter_type == "Average":
        return cv2.blur(img_np, (5, 5))
    elif filter_type == "Gaussian":
        return cv2.GaussianBlur(img_np, (5, 5), 1.0)
    elif filter_type == "Median":
        return cv2.medianBlur(img_np, 5)
    elif filter_type == "Sharpening":
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]], dtype=np.float32)
        return np.clip(cv2.filter2D(img_np, -1, kernel), 0, 255).astype(np.uint8)
    elif filter_type == "Sobel":
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        norm_mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.cvtColor(norm_mag, cv2.COLOR_GRAY2RGB)
    else:
        return img_np

# ----------------- DATASET PREPARATION -----------------
def find_data_root():
    cache_base = Path.home() / ".cache" / "kagglehub" / "datasets" / "nodoubttome" / "skin-cancer9-classesisic"
    if cache_base.exists():
        for p in cache_base.rglob("*"):
            if p.is_dir() and "train" in p.name.lower():
                subdirs = [x.name.lower() for x in p.iterdir() if x.is_dir()]
                if any(c in subdirs for c in HAM_CLASSES):
                    return p
    raise FileNotFoundError("HAM10000 dataset directory not found.")

def load_dataset_samples(data_root):
    data_by_class = defaultdict(list)
    for class_idx, class_name in enumerate(HAM_CLASSES):
        class_folder = data_root / class_name
        if not class_folder.exists():
            for d in data_root.iterdir():
                if d.is_dir() and class_name in d.name.lower():
                    class_folder = d
                    break
        files = list(class_folder.glob("*.jpg")) + list(class_folder.glob("*.jpeg")) + list(class_folder.glob("*.png"))
        data_by_class[class_name] = files

    rng = np.random.default_rng(SEED)
    train_samples, test_samples = [], []
    class_counts = {}

    for class_idx, class_name in enumerate(HAM_CLASSES):
        files = data_by_class[class_name]
        class_counts[class_name] = len(files)
        chosen = rng.choice(files, min(len(files), MAX_PER_CLASS), replace=False)
        n_tr = int(0.8 * len(chosen))
        for f in chosen[:n_tr]:
            train_samples.append((str(f), class_idx))
        for f in chosen[n_tr:]:
            test_samples.append((str(f), class_idx))

    return train_samples, test_samples, class_counts

# ----------------- MODEL LOADER -----------------
class FeatureBackbone(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name
        if name == "ResNet50":
            m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.features = nn.Sequential(*list(m.children())[:-2])
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.out_dim = 2048
        elif name == "ResNet101":
            m = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
            self.features = nn.Sequential(*list(m.children())[:-2])
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.out_dim = 2048
        elif name == "VGG16":
            m = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
            self.features = m.features
            self.pool = m.avgpool
            self.out_dim = 512 * 7 * 7

        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x):
        feat = self.features(x)
        feat = self.pool(feat)
        return torch.flatten(feat, 1)

# ----------------- FEATURE EXTRACTION -----------------
NORMALIZE = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

def extract_features_with_filter(model, samples, filter_type):
    model.eval()
    feats, labels = [], []
    batch_tensors, batch_labels = [], []

    for path, lbl in samples:
        img = Image.open(path).convert('RGB')
        img_np = np.array(img.resize((IMG_SIZE, IMG_SIZE)))
        filtered_np = apply_filter(img_np, filter_type)
        t = transforms.functional.to_tensor(filtered_np)
        t = NORMALIZE(t)
        batch_tensors.append(t)
        batch_labels.append(lbl)

        if len(batch_tensors) == 32:
            bx = torch.stack(batch_tensors).to(DEVICE)
            with torch.no_grad():
                f = model(bx)
            feats.append(f.cpu())
            labels.extend(batch_labels)
            batch_tensors, batch_labels = [], []

    if batch_tensors:
        bx = torch.stack(batch_tensors).to(DEVICE)
        with torch.no_grad():
            f = model(bx)
        feats.append(f.cpu())
        labels.extend(batch_labels)

    return torch.cat(feats, dim=0), torch.tensor(labels, dtype=torch.long)

# ----------------- HEAD CLASSIFIER -----------------
def train_and_eval_head(in_dim, X_tr, y_tr, X_te, y_te, epochs=15):
    clf = nn.Sequential(
        nn.Dropout(0.25),
        nn.Linear(in_dim, NUM_CLASSES)
    ).to(DEVICE)
    optimizer = optim.Adam(clf.parameters(), lr=0.0015, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    ds_tr = TensorDataset(X_tr, y_tr)
    loader_tr = DataLoader(ds_tr, batch_size=32, shuffle=True)

    tr_losses, tr_accs = [], []
    for ep in range(epochs):
        clf.train()
        total_loss, correct, total = 0.0, 0, 0
        for bx, by in loader_tr:
            optimizer.zero_grad()
            out = clf(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(by)
            correct += (out.argmax(1) == by).sum().item()
            total += len(by)
        tr_losses.append(total_loss / total)
        tr_accs.append(correct / total * 100.0)

    clf.eval()
    with torch.no_grad():
        logits = clf(X_te)
        probs = torch.softmax(logits, dim=1).numpy()
        preds = np.argmax(probs, axis=1)

    y_true = y_te.numpy()
    acc = accuracy_score(y_true, preds) * 100.0
    prec = precision_score(y_true, preds, average="macro", zero_division=0) * 100.0
    rec = recall_score(y_true, preds, average="macro", zero_division=0) * 100.0
    f1 = f1_score(y_true, preds, average="macro", zero_division=0) * 100.0
    macro_f1 = f1
    bal_acc = balanced_accuracy_score(y_true, preds) * 100.0

    try:
        auc = roc_auc_score(y_true, probs, multi_class="ovr", average="macro") * 100.0
    except Exception:
        auc = 0.0

    cm = confusion_matrix(y_true, preds, labels=list(range(NUM_CLASSES)))

    return {
        "accuracy": round(acc, 2),
        "precision": round(prec, 2),
        "recall": round(rec, 2),
        "f1": round(f1, 2),
        "macro_f1": round(macro_f1, 2),
        "balanced_acc": round(bal_acc, 2),
        "auc": round(auc, 2),
        "confusion_matrix": cm.tolist(),
        "train_loss": tr_losses,
        "train_acc": tr_accs
    }

# ----------------- VISUALIZATIONS -----------------
def generate_visualizations(data_root, train_samples, class_counts, results):
    print("Generating figures...")

    # 1. HAM10000 Class Distribution
    plt.figure(figsize=(9, 4.5))
    short_names = [c.title() for c in HAM_CLASSES]
    counts = [class_counts[c] for c in HAM_CLASSES]
    bars = plt.bar(short_names, counts, color='#2c7bb6', edgecolor='black', alpha=0.85)
    plt.title("HAM10000 Selected Lesion Class Distribution", fontsize=13, weight='bold')
    plt.xlabel("Diagnostic Class", fontsize=11)
    plt.ylabel("Available Images Count", fontsize=11)
    plt.xticks(rotation=25, ha='right')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 5, f"{int(yval)}", ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "class_distribution.png", dpi=200)
    plt.close()

    # 2. Original and Filtered Image Examples
    sample_img_path = train_samples[0][0]
    sample_img = Image.open(sample_img_path).convert('RGB')
    sample_np = np.array(sample_img.resize((IMG_SIZE, IMG_SIZE)))

    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    axes = axes.flatten()
    for idx, ftype in enumerate(FILTER_NAMES):
        filt_img = apply_filter(sample_np, ftype)
        axes[idx].imshow(filt_img)
        axes[idx].set_title(f"{ftype}", fontsize=12, weight='bold')
        axes[idx].axis('off')
    plt.suptitle("Dermoscopic Lesion Transformation Across Spatial Filters", fontsize=14, weight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "filtered_examples.png", dpi=200)
    plt.close()

    # 3. Model Comparison Bar Chart (Accuracy by Filter)
    plt.figure(figsize=(10, 5))
    x = np.arange(len(FILTER_NAMES))
    width = 0.26
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for idx, model_name in enumerate(BASE_MODELS):
        accs = [results[model_name][f]["accuracy"] for f in FILTER_NAMES]
        plt.bar(x + idx * width, accs, width=width, label=model_name, color=colors[idx], edgecolor='black', alpha=0.88)

    plt.title("Classification Accuracy (%) Across Spatial Filtering Conditions", fontsize=13, weight='bold')
    plt.xlabel("Applied Filter", fontsize=11)
    plt.ylabel("Test Accuracy (%)", fontsize=11)
    plt.xticks(x + width, FILTER_NAMES, fontsize=10)
    plt.legend(frameon=True)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_comparison_bar.png", dpi=200)
    plt.close()

    # 4. Confusion Matrices (Baseline vs Sharpening vs Sobel for ResNet50)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    selected_filters = ["No Filter", "Sharpening", "Sobel"]
    for idx, ftype in enumerate(selected_filters):
        cm = np.array(results["ResNet50"][ftype]["confusion_matrix"])
        im = axes[idx].imshow(cm, cmap='Blues', interpolation='nearest')
        axes[idx].set_title(f"ResNet50 ({ftype})", fontsize=11, weight='bold')
        axes[idx].set_xlabel("Predicted Label", fontsize=9)
        axes[idx].set_ylabel("True Label", fontsize=9)
        fig.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
    plt.suptitle("Confusion Matrix Comparison on HAM10000 Test Set", fontsize=13, weight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrices.png", dpi=200)
    plt.close()

    # 5. Training Loss & Accuracy Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for ftype, col in zip(["No Filter", "Gaussian", "Sharpening", "Sobel"], ['black', 'blue', 'green', 'red']):
        loss = results["ResNet50"][ftype]["train_loss"]
        acc = results["ResNet50"][ftype]["train_acc"]
        ax1.plot(loss, label=ftype, color=col, linewidth=2)
        ax2.plot(acc, label=ftype, color=col, linewidth=2)

    ax1.set_title("Training Loss (ResNet50 Head)", fontsize=11, weight='bold')
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Cross Entropy Loss", fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2.set_title("Training Accuracy (ResNet50 Head)", fontsize=11, weight='bold')
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Accuracy (%)", fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.suptitle("Learning Dynamics Under Different Spatial Filtering Conditions", fontsize=12, weight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "learning_curves.png", dpi=200)
    plt.close()
    print("All figures successfully created.")

# ----------------- MAIN PIPELINE -----------------
def main():
    print("=" * 60)
    print("Lab Task 02: Effect of Image Filtering on Skin Lesion Classification")
    print("=" * 60)

    data_root = find_data_root()
    print(f"Data root: {data_root}")
    train_samples, test_samples, class_counts = load_dataset_samples(data_root)
    print(f"Loaded: {len(train_samples)} Train samples | {len(test_samples)} Test samples across {NUM_CLASSES} classes")

    results = {m: {} for m in BASE_MODELS}

    for model_name in BASE_MODELS:
        print(f"\n==================== [{model_name}] ====================")
        backbone = FeatureBackbone(model_name).to(DEVICE)

        for filter_type in FILTER_NAMES:
            t0 = time.time()
            X_tr, y_tr = extract_features_with_filter(backbone, train_samples, filter_type)
            X_te, y_te = extract_features_with_filter(backbone, test_samples, filter_type)
            metrics = train_and_eval_head(backbone.out_dim, X_tr, y_tr, X_te, y_te, epochs=15)
            results[model_name][filter_type] = metrics
            elapsed = time.time() - t0
            print(f"  {filter_type:<12} -> Acc: {metrics['accuracy']:>5.2f}% | F1: {metrics['f1']:>5.2f}% | BalAcc: {metrics['balanced_acc']:>5.2f}% | AUC: {metrics['auc']:>5.2f}% ({elapsed:.1f}s)")

    # Save results JSON
    with open(RESULTS_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved raw results to {RESULTS_DIR / 'results.json'}")

    # Generate charts and plots
    generate_visualizations(data_root, train_samples, class_counts, results)

    # Print Table
    print("\n" + "=" * 80)
    print(f"{'Model':<12} {'Filter':<12} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-score':<10} {'Macro-F1':<10} {'AUC':<10}")
    print("-" * 80)
    for model_name in BASE_MODELS:
        for filter_type in FILTER_NAMES:
            m = results[model_name][filter_type]
            print(f"{model_name:<12} {filter_type:<12} {m['accuracy']:>8.2f}% {m['precision']:>8.2f}% {m['recall']:>8.2f}% {m['f1']:>8.2f}% {m['macro_f1']:>8.2f}% {m['auc']:>8.2f}%")

    print("=" * 80)
    print("All experiments completed successfully.")

if __name__ == "__main__":
    main()
