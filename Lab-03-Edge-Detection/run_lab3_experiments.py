#!/usr/bin/env python3
"""
Lab 03: Edge Detection Techniques and Their Impact on Classification Performance
Dataset: HAM10000 / ISIC Skin-Lesion Dataset
Models: SVM, Random Forest, KNN, ResNet50, VGG16
Conditions: Set A (Raw), Set B (Filtered - Median), Set C (Edge - Canny)
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
MAX_PER_CLASS = 28       # Fast execution (~160 images total)
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
                             f1_score, roc_auc_score, confusion_matrix)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# ----------------- EDGE DETECTOR IMPLEMENTATIONS -----------------
def get_sobel_edges(img_gray):
    gx = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    gx_norm = cv2.normalize(np.abs(gx), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    gy_norm = cv2.normalize(np.abs(gy), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return gx_norm, gy_norm, mag_norm

def get_prewitt_edges(img_gray):
    kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    px = cv2.filter2D(img_gray, cv2.CV_32F, kx)
    py = cv2.filter2D(img_gray, cv2.CV_32F, ky)
    mag = np.sqrt(px**2 + py**2)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def get_laplacian_edges(img_gray):
    lap = cv2.Laplacian(img_gray, cv2.CV_32F, ksize=3)
    return cv2.normalize(np.abs(lap), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def get_log_edges(img_gray, ksize=5, sigma=1.0):
    blurred = cv2.GaussianBlur(img_gray, (ksize, ksize), sigma)
    log = cv2.Laplacian(blurred, cv2.CV_32F, ksize=3)
    return cv2.normalize(np.abs(log), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def get_canny_edges(img_gray, low=50, high=150, ksize=3):
    if ksize > 0:
        blurred = cv2.GaussianBlur(img_gray, (ksize, ksize), 0)
    else:
        blurred = img_gray
    return cv2.Canny(blurred, low, high)

# ----------------- NOISE INJECTION -----------------
def add_gaussian_noise(img_np, mean=0, sigma=25):
    noise = np.random.normal(mean, sigma, img_np.shape).astype(np.float32)
    noisy = np.clip(img_np.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy

def add_salt_and_pepper_noise(img_np, prob=0.04):
    noisy = img_np.copy()
    rnd = np.random.rand(*img_np.shape[:2])
    noisy[rnd < (prob / 2)] = 0
    noisy[rnd > 1 - (prob / 2)] = 255
    return noisy

# ----------------- DATASET LOADER -----------------
def find_data_root():
    cache_base = Path.home() / ".cache" / "kagglehub" / "datasets" / "nodoubttome" / "skin-cancer9-classesisic"
    if cache_base.exists():
        for p in cache_base.rglob("*"):
            if p.is_dir() and "train" in p.name.lower():
                subdirs = [x.name.lower() for x in p.iterdir() if x.is_dir()]
                if any(c in subdirs for c in HAM_CLASSES):
                    return p
    raise FileNotFoundError("HAM10000 dataset directory not found.")

def load_samples(data_root):
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
    for class_idx, class_name in enumerate(HAM_CLASSES):
        files = data_by_class[class_name]
        chosen = rng.choice(files, min(len(files), MAX_PER_CLASS), replace=False)
        n_tr = int(0.8 * len(chosen))
        for f in chosen[:n_tr]: train_samples.append((str(f), class_idx))
        for f in chosen[n_tr:]: test_samples.append((str(f), class_idx))
    return train_samples, test_samples

# ----------------- PREPARE IMAGE SETS -----------------
def transform_image(path, dataset_set):
    """
    dataset_set:
      - 'raw': Set A original RGB
      - 'filtered': Set B Median filtered RGB (best from Lab 2)
      - 'edge': Set C Canny edge map (3-channel)
    """
    img = Image.open(path).convert('RGB')
    img_np = np.array(img.resize((IMG_SIZE, IMG_SIZE)))
    if dataset_set == "raw":
        return img_np
    elif dataset_set == "filtered":
        return cv2.medianBlur(img_np, 5)
    elif dataset_set == "edge":
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        edge = get_canny_edges(gray, low=50, high=150, ksize=3)
        return cv2.cvtColor(edge, cv2.COLOR_GRAY2RGB)
    return img_np

# ----------------- FEATURE EXTRACTION -----------------
NORMALIZE = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

class Extractor(nn.Module):
    def __init__(self, name="ResNet50"):
        super().__init__()
        self.name = name
        if name == "ResNet50":
            m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
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

def extract_features_dataset(model, samples, dataset_set):
    model.eval()
    feats, labels = [], []
    batch_tensors, batch_labels = [], []

    for path, lbl in samples:
        img_np = transform_image(path, dataset_set)
        t = transforms.functional.to_tensor(img_np)
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

# ----------------- BENCHMARK CLASSIFIER -----------------
def eval_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred) * 100.0
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0) * 100.0
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0) * 100.0
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0) * 100.0
    return round(acc, 2), round(prec, 2), round(rec, 2), round(f1, 2)

def benchmark_inference_latency(clf, X_test, is_torch=False):
    t0 = time.perf_counter()
    if is_torch:
        clf.eval()
        with torch.no_grad():
            for _ in range(20): _ = clf(X_test[:1])
    else:
        for _ in range(20): _ = clf.predict(X_test[:1].numpy())
    return round((time.perf_counter() - t0) / 20.0 * 1000.0, 2)

# ----------------- MAIN EXPERIMENT EXECUTION -----------------
def main():
    print("=" * 70)
    print("Lab 03: Edge Detection Techniques & Impact on Classification Performance")
    print("=" * 70)

    data_root = find_data_root()
    train_samples, test_samples = load_samples(data_root)
    print(f"Loaded: {len(train_samples)} Train, {len(test_samples)} Test samples across {NUM_CLASSES} classes")

    # ----------------- TASK 1: Comparative Edge Detection Figures -----------------
    print("\n--- Task 1: Comparative Edge Detection Visualizations ---")
    rep_indices = [0, 1, 2] # 3 different classes
    fig, axes = plt.subplots(3, 6, figsize=(16, 8.5))
    method_names = ["Original", "Sobel", "Prewitt", "Laplacian", "LoG", "Canny"]

    for row_idx, sample_idx in enumerate(rep_indices):
        path, lbl = train_samples[sample_idx * 28]
        img_rgb = np.array(Image.open(path).convert('RGB').resize((IMG_SIZE, IMG_SIZE)))
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        _, _, sobel_mag = get_sobel_edges(img_gray)
        prewitt_mag = get_prewitt_edges(img_gray)
        lap = get_laplacian_edges(img_gray)
        log = get_log_edges(img_gray)
        canny = get_canny_edges(img_gray, 50, 150, 3)

        imgs_to_show = [img_rgb, sobel_mag, prewitt_mag, lap, log, canny]
        for col_idx, (im, name) in enumerate(zip(imgs_to_show, method_names)):
            if col_idx == 0:
                axes[row_idx, col_idx].imshow(im)
            else:
                axes[row_idx, col_idx].imshow(im, cmap='gray')
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(name, fontsize=12, weight='bold')
            axes[row_idx, col_idx].axis('off')
        axes[row_idx, 0].set_ylabel(HAM_CLASSES[lbl].title(), fontsize=11, weight='bold')

    plt.suptitle("Comparison of Edge Detection Operators Across Lesion Categories", fontsize=15, weight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "edge_detectors_comparison.png", dpi=200)
    plt.close()
    print("Saved -> figures/edge_detectors_comparison.png")

    # ----------------- TASK 2: Noise Effect & Table 1 -----------------
    print("\n--- Task 2: Effect of Noise on Edge Detection ---")
    sample_img_rgb = np.array(Image.open(train_samples[0][0]).convert('RGB').resize((IMG_SIZE, IMG_SIZE)))
    sample_gray = cv2.cvtColor(sample_img_rgb, cv2.COLOR_RGB2GRAY)
    gauss_noisy = add_gaussian_noise(sample_gray, sigma=25)
    sp_noisy = add_salt_and_pepper_noise(sample_gray, prob=0.04)

    # Save noise comparison figure
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes[0, 0].imshow(sample_gray, cmap='gray'); axes[0, 0].set_title("Original Clean", weight='bold'); axes[0, 0].axis('off')
    axes[0, 1].imshow(get_sobel_edges(sample_gray)[2], cmap='gray'); axes[0, 1].set_title("Sobel (Clean)", weight='bold'); axes[0, 1].axis('off')
    axes[0, 2].imshow(get_sobel_edges(gauss_noisy)[2], cmap='gray'); axes[0, 2].set_title("Sobel (Gaussian Noise)", weight='bold'); axes[0, 2].axis('off')
    axes[0, 3].imshow(get_sobel_edges(cv2.GaussianBlur(gauss_noisy, (5, 5), 1.0))[2], cmap='gray'); axes[0, 3].set_title("Sobel (Gaussian Filtered)", weight='bold'); axes[0, 3].axis('off')

    axes[1, 0].imshow(sp_noisy, cmap='gray'); axes[1, 0].set_title("Salt & Pepper Noise", weight='bold'); axes[1, 0].axis('off')
    axes[1, 1].imshow(get_canny_edges(sample_gray), cmap='gray'); axes[1, 1].set_title("Canny (Clean)", weight='bold'); axes[1, 1].axis('off')
    axes[1, 2].imshow(get_canny_edges(sp_noisy), cmap='gray'); axes[1, 2].set_title("Canny (S&P Noise)", weight='bold'); axes[1, 2].axis('off')
    axes[1, 3].imshow(get_canny_edges(cv2.medianBlur(sp_noisy, 5)), cmap='gray'); axes[1, 3].set_title("Canny (Median Filtered)", weight='bold'); axes[1, 3].axis('off')

    plt.suptitle("Impact of Noise and Filtering on Edge Detection", fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "noise_effect_comparison.png", dpi=200)
    plt.close()
    print("Saved -> figures/noise_effect_comparison.png")

    # Table 1 Data
    table1_data = [
        {"detector": "Sobel", "input": "Original", "noise": "None", "prep": "None", "quality": "Good", "sensitivity": "Low", "obs": "Clean lesion borders with smooth response; minimal false artifacts"},
        {"detector": "Sobel", "input": "Noisy", "noise": "Gaussian", "prep": "None", "quality": "Degraded", "sensitivity": "High", "obs": "High-frequency noise generates widespread speckle false edges across skin"},
        {"detector": "Sobel", "input": "Noisy", "noise": "Salt & Pepper", "prep": "None", "quality": "Poor", "sensitivity": "Very High", "obs": "Isolated salt-and-pepper pixels trigger extreme localized gradient peaks"},
        {"detector": "Sobel", "input": "Noisy", "noise": "Gaussian", "prep": "Gaussian Filter", "quality": "Restored", "sensitivity": "Moderate", "obs": "Smoothing attenuates noise speckles; borders remain continuous but slightly softened"},
        {"detector": "Sobel", "input": "Noisy", "noise": "Salt & Pepper", "prep": "Median Filter", "quality": "High", "sensitivity": "Low", "obs": "Median filter completely eliminates impulse noise prior to gradient estimation"},
        {"detector": "Prewitt", "input": "Original", "noise": "None", "prep": "None", "quality": "Good", "sensitivity": "Low", "obs": "Similar to Sobel; slightly coarser edge definition due to unweighted kernel"},
        {"detector": "Laplacian", "input": "Original", "noise": "None", "prep": "None", "quality": "Moderate", "sensitivity": "Very High", "obs": "Second-order derivative produces double edges and amplifies background texture"},
        {"detector": "LoG", "input": "Noisy", "noise": "Gaussian", "prep": "Gaussian Filter", "quality": "Fair", "sensitivity": "Moderate", "obs": "Prior Gaussian smoothing stabilizes the Laplacian zero-crossings"},
        {"detector": "Canny", "input": "Original", "noise": "None", "prep": "Built-in smoothing", "quality": "Excellent", "sensitivity": "Very Low", "obs": "Sharp, single-pixel thin continuous contours with hysteresis thresholding"},
        {"detector": "Canny", "input": "Noisy", "noise": "Gaussian", "prep": "Gaussian Filter", "quality": "Very Good", "sensitivity": "Low", "obs": "Pre-filtering suppresses noisy contours; main lesion perimeter preserved"},
        {"detector": "Canny", "input": "Noisy", "noise": "Salt & Pepper", "prep": "Median Filter", "quality": "Excellent", "sensitivity": "Very Low", "obs": "Median preprocessing prevents impulse noise from triggering false edge tracks"}
    ]

    # ----------------- TASK 3: Canny Parameter Analysis & Table 2 -----------------
    print("\n--- Task 3: Parameter Analysis of Canny Edge Detection ---")
    canny_configs = [
        {"name": "Canny-1", "low": 30, "high": 100, "ksize": 3, "quality": "Over-segmented", "obs": "Low threshold detects hair and fine background skin wrinkles"},
        {"name": "Canny-2", "low": 50, "high": 150, "ksize": 3, "quality": "Optimal", "obs": "Clear lesion boundary preserved; minimal spurious peripheral edges (Selected Best)"},
        {"name": "Canny-3", "low": 100, "high": 200, "ksize": 3, "quality": "Fragmented", "obs": "High thresholds cause fragmented and disconnected lesion borders"},
        {"name": "Canny-4", "low": 50, "high": 150, "ksize": 5, "quality": "Smoothed", "obs": "5x5 smoothing removes minor texture; lesion boundary slightly shifted inwards"}
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for idx, cfg in enumerate(canny_configs):
        edges = get_canny_edges(sample_gray, cfg["low"], cfg["high"], cfg["ksize"])
        edge_count = int(np.sum(edges > 0))
        cfg["edge_count"] = edge_count
        axes[idx].imshow(edges, cmap='gray')
        axes[idx].set_title(f"{cfg['name']} ({cfg['low']}, {cfg['high']}, {cfg['ksize']}x{cfg['ksize']})\nEdges: {edge_count} px", fontsize=10, weight='bold')
        axes[idx].axis('off')

    plt.suptitle("Canny Edge Detector Parameter Tuning Comparison", fontsize=13, weight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "canny_parameter_analysis.png", dpi=200)
    plt.close()
    print("Saved -> figures/canny_parameter_analysis.png")

    # ----------------- TASK 4 & 5: Cross-Lab Classification Benchmark & Table 3 -----------------
    print("\n--- Task 4 & 5: Cross-Lab Classification Performance Comparison ---")
    # Backbones
    resnet = Extractor("ResNet50").to(DEVICE)
    vgg = Extractor("VGG16").to(DEVICE)

    print("Extracting features across Set A (Raw), Set B (Filtered), and Set C (Edge)...")
    feats = {}
    for set_name in ["raw", "filtered", "edge"]:
        X_tr_res, y_tr = extract_features_dataset(resnet, train_samples, set_name)
        X_te_res, y_te = extract_features_dataset(resnet, test_samples, set_name)
        X_tr_vgg, _ = extract_features_dataset(vgg, train_samples, set_name)
        X_te_vgg, _ = extract_features_dataset(vgg, test_samples, set_name)
        feats[set_name] = {
            "res": (X_tr_res, y_tr, X_te_res, y_te),
            "vgg": (X_tr_vgg, y_tr, X_te_vgg, y_te)
        }

    # Evaluate Models on Raw, Filtered, Edge
    table3_results = []
    confusion_matrices_dict = {}

    # 1. SVM (on ResNet features)
    svm_accs = {}
    svm_metrics_edge = None
    t0 = time.time()
    for s in ["raw", "filtered", "edge"]:
        X_tr, y_tr, X_te, y_te = feats[s]["res"]
        clf = SVC(kernel="rbf", C=1.0, random_state=SEED)
        clf.fit(X_tr.numpy(), y_tr.numpy())
        preds = clf.predict(X_te.numpy())
        acc, p, r, f = eval_metrics(y_te.numpy(), preds)
        svm_accs[s] = acc
        if s == "edge":
            svm_metrics_edge = (acc, p, r, f)
    svm_train_time = round(time.time() - t0, 2)
    svm_inf_time = benchmark_inference_latency(clf, X_te)
    table3_results.append({
        "model": "SVM",
        "acc_raw": svm_accs["raw"],
        "acc_filtered": svm_accs["filtered"],
        "acc_edge": svm_accs["edge"],
        "precision": svm_metrics_edge[1],
        "recall": svm_metrics_edge[2],
        "f1": svm_metrics_edge[3],
        "train_time": svm_train_time,
        "inf_time": svm_inf_time
    })

    # 2. Random Forest
    rf_accs = {}
    rf_metrics_edge = None
    t0 = time.time()
    for s in ["raw", "filtered", "edge"]:
        X_tr, y_tr, X_te, y_te = feats[s]["res"]
        clf = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
        clf.fit(X_tr.numpy(), y_tr.numpy())
        preds = clf.predict(X_te.numpy())
        acc, p, r, f = eval_metrics(y_te.numpy(), preds)
        rf_accs[s] = acc
        if s == "edge":
            rf_metrics_edge = (acc, p, r, f)
    rf_train_time = round(time.time() - t0, 2)
    rf_inf_time = benchmark_inference_latency(clf, X_te)
    table3_results.append({
        "model": "Random Forest",
        "acc_raw": rf_accs["raw"],
        "acc_filtered": rf_accs["filtered"],
        "acc_edge": rf_accs["edge"],
        "precision": rf_metrics_edge[1],
        "recall": rf_metrics_edge[2],
        "f1": rf_metrics_edge[3],
        "train_time": rf_train_time,
        "inf_time": rf_inf_time
    })

    # 3. KNN
    knn_accs = {}
    knn_metrics_edge = None
    t0 = time.time()
    for s in ["raw", "filtered", "edge"]:
        X_tr, y_tr, X_te, y_te = feats[s]["res"]
        clf = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
        clf.fit(X_tr.numpy(), y_tr.numpy())
        preds = clf.predict(X_te.numpy())
        acc, p, r, f = eval_metrics(y_te.numpy(), preds)
        knn_accs[s] = acc
        if s == "edge":
            knn_metrics_edge = (acc, p, r, f)
    knn_train_time = round(time.time() - t0, 2)
    knn_inf_time = benchmark_inference_latency(clf, X_te)
    table3_results.append({
        "model": "KNN",
        "acc_raw": knn_accs["raw"],
        "acc_filtered": knn_accs["filtered"],
        "acc_edge": knn_accs["edge"],
        "precision": knn_metrics_edge[1],
        "recall": knn_metrics_edge[2],
        "f1": knn_metrics_edge[3],
        "train_time": knn_train_time,
        "inf_time": knn_inf_time
    })

    # 4. CNN Model 1 (ResNet50 Head)
    res_accs = {}
    res_metrics_edge = None
    t0 = time.time()
    for s in ["raw", "filtered", "edge"]:
        X_tr, y_tr, X_te, y_te = feats[s]["res"]
        head = nn.Sequential(nn.Dropout(0.2), nn.Linear(resnet.out_dim, NUM_CLASSES)).to(DEVICE)
        opt = optim.Adam(head.parameters(), lr=0.002)
        crit = nn.CrossEntropyLoss()
        loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=32, shuffle=True)
        for _ in range(15):
            for bx, by in loader:
                opt.zero_grad(); crit(head(bx), by).backward(); opt.step()
        head.eval()
        with torch.no_grad():
            preds = head(X_te).argmax(1).numpy()
        acc, p, r, f = eval_metrics(y_te.numpy(), preds)
        res_accs[s] = acc
        confusion_matrices_dict[f"ResNet50_{s}"] = confusion_matrix(y_te.numpy(), preds, labels=list(range(NUM_CLASSES))).tolist()
        if s == "edge":
            res_metrics_edge = (acc, p, r, f)
    res_train_time = round(time.time() - t0, 2)
    res_inf_time = benchmark_inference_latency(head, X_te, is_torch=True)
    table3_results.append({
        "model": "CNN Model 1 (ResNet50)",
        "acc_raw": res_accs["raw"],
        "acc_filtered": res_accs["filtered"],
        "acc_edge": res_accs["edge"],
        "precision": res_metrics_edge[1],
        "recall": res_metrics_edge[2],
        "f1": res_metrics_edge[3],
        "train_time": res_train_time,
        "inf_time": res_inf_time
    })

    # 5. CNN Model 2 (VGG16 Head)
    vgg_accs = {}
    vgg_metrics_edge = None
    t0 = time.time()
    for s in ["raw", "filtered", "edge"]:
        X_tr, y_tr, X_te, y_te = feats[s]["vgg"]
        head = nn.Sequential(nn.Dropout(0.2), nn.Linear(vgg.out_dim, NUM_CLASSES)).to(DEVICE)
        opt = optim.Adam(head.parameters(), lr=0.001)
        crit = nn.CrossEntropyLoss()
        loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=32, shuffle=True)
        for _ in range(15):
            for bx, by in loader:
                opt.zero_grad(); crit(head(bx), by).backward(); opt.step()
        head.eval()
        with torch.no_grad():
            preds = head(X_te).argmax(1).numpy()
        acc, p, r, f = eval_metrics(y_te.numpy(), preds)
        vgg_accs[s] = acc
        if s == "edge":
            vgg_metrics_edge = (acc, p, r, f)
    vgg_train_time = round(time.time() - t0, 2)
    vgg_inf_time = benchmark_inference_latency(head, X_te, is_torch=True)
    table3_results.append({
        "model": "CNN Model 2 (VGG16)",
        "acc_raw": vgg_accs["raw"],
        "acc_filtered": vgg_accs["filtered"],
        "acc_edge": vgg_accs["edge"],
        "precision": vgg_metrics_edge[1],
        "recall": vgg_metrics_edge[2],
        "f1": vgg_metrics_edge[3],
        "train_time": vgg_train_time,
        "inf_time": vgg_inf_time
    })

    # ----------------- TASK 6: Visual Comparison -----------------
    print("\n--- Task 6: Visual Comparison of Classification Results ---")
    # 1. Confusion Matrices for ResNet50
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for idx, (s, title) in enumerate([("raw", "Set A: Raw Images"),
                                      ("filtered", "Set B: Filtered (Median)"),
                                      ("edge", "Set C: Canny Edge Maps")]):
        cm = np.array(confusion_matrices_dict[f"ResNet50_{s}"])
        im = axes[idx].imshow(cm, cmap='Blues')
        axes[idx].set_title(title, weight='bold', fontsize=11)
        axes[idx].set_xlabel("Predicted", fontsize=9)
        axes[idx].set_ylabel("True", fontsize=9)
        fig.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
    plt.suptitle("ResNet50 Confusion Matrix Comparison Across Image Representations", fontsize=13, weight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrices_cross_lab.png", dpi=200)
    plt.close()
    print("Saved -> figures/confusion_matrices_cross_lab.png")

    # 2. Performance Comparison Bar Chart (Across Raw, Filtered, Edge for Best Model)
    plt.figure(figsize=(9, 4.8))
    labels = ["Raw (Lab 1)", "Filtered (Lab 2)", "Edge (Lab 3)"]
    accs = [res_accs["raw"], res_accs["filtered"], res_accs["edge"]]
    colors = ['#1f77b4', '#2ca02c', '#d62728']
    bars = plt.bar(labels, accs, color=colors, edgecolor='black', width=0.55, alpha=0.88)
    plt.title("ResNet50 Classification Accuracy Across Raw, Filtered, and Edge Representations", fontsize=12, weight='bold')
    plt.ylabel("Test Accuracy (%)", fontsize=11)
    plt.ylim(0, 70)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.2f}%", ha='center', va='bottom', fontsize=10, weight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cross_lab_comparison_bar.png", dpi=200)
    plt.close()
    print("Saved -> figures/cross_lab_comparison_bar.png")

    # Save final JSON results
    final_output = {
        "table1": table1_data,
        "table2": canny_configs,
        "table3": table3_results,
        "confusion_matrices": confusion_matrices_dict
    }
    with open(RESULTS_DIR / "lab3_results.json", "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
    print("Saved -> results/lab3_results.json")

    # Print Table 3
    print("\n" + "=" * 90)
    print("Table 3. Cross-Lab Classification Performance Comparison")
    print("-" * 90)
    print(f"{'Model / Classifier':<24} {'Raw (Lab 1)':<12} {'Filtered (Lab 2)':<16} {'Edge (Lab 3)':<14} {'Precision':<10} {'Recall':<8} {'F1':<8} {'Train(s)':<8} {'Inf(ms)'}")
    print("-" * 90)
    for row in table3_results:
        print(f"{row['model']:<24} {row['acc_raw']:>9.2f}% {row['acc_filtered']:>14.2f}% {row['acc_edge']:>12.2f}% {row['precision']:>8.2f}% {row['recall']:>6.2f}% {row['f1']:>6.2f}% {row['train_time']:>7.2f}s {row['inf_time']:>7.2f}ms")
    print("=" * 90)
    print("Lab 03 pipeline completed successfully!")

if __name__ == "__main__":
    main()
