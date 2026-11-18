#!/usr/bin/env python3
"""
generate_notebook.py
Generates the Google Colab compatible Jupyter Notebook: Lab2_Skin_Lesion_Filtering.ipynb
"""
import json
from pathlib import Path

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Lab Task 02: Effect of Image Filtering on Skin-Lesion Classification\n",
    "\n",
    "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TheBilalButt/lab-02-image-filtering/blob/main/Lab2_Skin_Lesion_Filtering.ipynb)\n",
    "\n",
    "**Course**: Computer Vision / Deep Learning Lab  \n",
    "**Author**: Bilal Butt ([@TheBilalButt](https://github.com/TheBilalButt))  \n",
    "**Dataset**: HAM10000 Skin-Lesion Dataset (ISIC 7-class subset)  \n",
    "**Models**: Top 3 Pretrained Models from Lab 01 (ResNet50, ResNet101, VGG16)  \n",
    "**Filters**: No Filter (Baseline), Average/Mean, Gaussian Blur, Median, Sharpening, Sobel Edge Filter"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Setup and Environment Configuration\n",
    "Install required dependencies and verify GPU/CPU runtime."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Dependencies installation (run in Colab)\n",
    "!pip install -q opencv-python kagglehub scikit-learn matplotlib seaborn torch torchvision pillow"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os, sys, json, time, warnings\n",
    "from pathlib import Path\n",
    "from collections import defaultdict\n",
    "import numpy as np\n",
    "import cv2\n",
    "from PIL import Image\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.optim as optim\n",
    "import torchvision.models as models\n",
    "import torchvision.transforms as transforms\n",
    "from torch.utils.data import DataLoader, TensorDataset\n",
    "from sklearn.metrics import (\n",
    "    accuracy_score, precision_score, recall_score,\n",
    "    f1_score, balanced_accuracy_score, roc_auc_score,\n",
    "    confusion_matrix, classification_report\n",
    ")\n",
    "\n",
    "warnings.filterwarnings('ignore')\n",
    "DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
    "print(f\"Using compute device: {DEVICE}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Dataset Preparation (HAM10000 Skin-Lesion)\n",
    "We select the 7 primary diagnostic categories from the HAM10000 / ISIC dataset:\n",
    "1. `actinic keratosis` (akiec)\n",
    "2. `basal cell carcinoma` (bcc)\n",
    "3. `dermatofibroma` (df)\n",
    "4. `melanoma` (mel)\n",
    "5. `nevus` (nv)\n",
    "6. `pigmented benign keratosis` (bkl)\n",
    "7. `vascular lesion` (vasc)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import kagglehub\n",
    "\n",
    "# Download or locate the dataset via kagglehub\n",
    "dataset_path = kagglehub.dataset_download(\"nodoubttome/skin-cancer9-classesisic\")\n",
    "print(\"Dataset root located at:\", dataset_path)\n",
    "\n",
    "HAM_CLASSES = [\n",
    "    'actinic keratosis',\n",
    "    'basal cell carcinoma',\n",
    "    'dermatofibroma',\n",
    "    'melanoma',\n",
    "    'nevus',\n",
    "    'pigmented benign keratosis',\n",
    "    'vascular lesion'\n",
    "]\n",
    "NUM_CLASSES = len(HAM_CLASSES)\n",
    "IMG_SIZE = 224"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Spatial-Domain Image Processing Filters\n",
    "Implementation of the five spatial-domain filters alongside the baseline:\n",
    "- **Average/Mean Filter**: 5x5 smoothing filter to reduce high-frequency noise.\n",
    "- **Gaussian Filter**: 5x5 Gaussian convolution kernel (sigma=1.0) with distance-weighted smoothing.\n",
    "- **Median Filter**: 5x5 non-linear rank filter effective against salt-and-pepper noise.\n",
    "- **Sharpening Filter**: 3x3 high-pass unsharp masking kernel emphasizing micro-textures.\n",
    "- **Sobel Edge Filter**: Discrete differentiation operator computing gradient magnitude."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def apply_filter(img_np, filter_type):\n",
    "    \"\"\"Apply spatial-domain transformation to an RGB NumPy image [0..255].\"\"\"\n",
    "    if filter_type == \"No Filter\":\n",
    "        return img_np\n",
    "    elif filter_type == \"Average\":\n",
    "        return cv2.blur(img_np, (5, 5))\n",
    "    elif filter_type == \"Gaussian\":\n",
    "        return cv2.GaussianBlur(img_np, (5, 5), 1.0)\n",
    "    elif filter_type == \"Median\":\n",
    "        return cv2.medianBlur(img_np, 5)\n",
    "    elif filter_type == \"Sharpening\":\n",
    "        kernel = np.array([[0, -1, 0],\n",
    "                           [-1, 5, -1],\n",
    "                           [0, -1, 0]], dtype=np.float32)\n",
    "        return np.clip(cv2.filter2D(img_np, -1, kernel), 0, 255).astype(np.uint8)\n",
    "    elif filter_type == \"Sobel\":\n",
    "        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)\n",
    "        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)\n",
    "        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)\n",
    "        mag = cv2.magnitude(gx, gy)\n",
    "        norm_mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)\n",
    "        return cv2.cvtColor(norm_mag, cv2.COLOR_GRAY2RGB)\n",
    "    return img_np"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Model Loading (Best 3 from Lab 01)\n",
    "Loading the top three models identified from Lab Activity 1:\n",
    "1. **ResNet50**\n",
    "2. **ResNet101**\n",
    "3. **VGG16**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "class FeatureBackbone(nn.Module):\n",
    "    def __init__(self, name):\n",
    "        super().__init__()\n",
    "        self.name = name\n",
    "        if name == \"ResNet50\":\n",
    "            m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)\n",
    "            self.features = nn.Sequential(*list(m.children())[:-2])\n",
    "            self.pool = nn.AdaptiveAvgPool2d((1, 1))\n",
    "            self.out_dim = 2048\n",
    "        elif name == \"ResNet101\":\n",
    "            m = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)\n",
    "            self.features = nn.Sequential(*list(m.children())[:-2])\n",
    "            self.pool = nn.AdaptiveAvgPool2d((1, 1))\n",
    "            self.out_dim = 2048\n",
    "        elif name == \"VGG16\":\n",
    "            m = models.vgg16(weights=models.VGG16_Weights.DEFAULT)\n",
    "            self.features = m.features\n",
    "            self.pool = m.avgpool\n",
    "            self.out_dim = 512 * 7 * 7\n",
    "        \n",
    "        for p in self.features.parameters():\n",
    "            p.requires_grad = False\n",
    "\n",
    "    def forward(self, x):\n",
    "        feat = self.features(x)\n",
    "        feat = self.pool(feat)\n",
    "        return torch.flatten(feat, 1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Execution Pipeline and Comparative Evaluation\n",
    "Run the 18 experimental combinations (3 architectures x 6 filtering conditions)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Execute experiments script locally or in notebook\n",
    "!python run_experiments.py"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Visualizations and Experimental Findings"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from IPython.display import Image, display\n",
    "\n",
    "print(\"1. HAM10000 Class Distribution:\")\n",
    "display(Image(\"figures/class_distribution.png\"))\n",
    "\n",
    "print(\"2. Dermoscopic Lesion Transformations Across Filters:\")\n",
    "display(Image(\"figures/filtered_examples.png\"))\n",
    "\n",
    "print(\"3. Accuracy Comparison Across Spatial Filters:\")\n",
    "display(Image(\"figures/model_comparison_bar.png\"))\n",
    "\n",
    "print(\"4. Confusion Matrices (Baseline vs Sharpening vs Sobel):\")\n",
    "display(Image(\"figures/confusion_matrices.png\"))\n",
    "\n",
    "print(\"5. Loss & Accuracy Learning Curves:\")\n",
    "display(Image(\"figures/learning_curves.png\"))"
   ]
  }
 ],
 "metadata": {
  "accelerator": "GPU",
  "colab": {
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open("Lab2_Skin_Lesion_Filtering.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("Generated Lab2_Skin_Lesion_Filtering.ipynb successfully.")
