#!/usr/bin/env python3
"""
generate_notebook.py
Generates the Google Colab compatible Jupyter Notebook: Lab3_Edge_Detection_Classification.ipynb
"""
import json
from pathlib import Path

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Lab 03: Edge Detection Techniques and Their Impact on Classification Performance\n",
    "\n",
    "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TheBilalButt/lab-03-edge-detection/blob/main/Lab3_Edge_Detection_Classification.ipynb)\n",
    "[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/TheBilalButt/lab-03-edge-detection)\n",
    "\n",
    "**Course**: Computer Vision / Deep Learning Lab  \n",
    "**Author**: Bilal Butt ([@TheBilalButt](https://github.com/TheBilalButt))  \n",
    "**Dataset**: HAM10000 / ISIC Skin-Lesion Dataset  \n",
    "**Techniques**: First-Order (Sobel, Prewitt), Second-Order (Laplacian, LoG), Multi-Stage (Canny)  \n",
    "**Cross-Lab Evaluation**: Set A (Raw), Set B (Filtered), Set C (Edge Maps)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Setup and Environment Configuration"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Install dependencies (for Google Colab)\n",
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
    "import torchvision.models as models\n",
    "import torchvision.transforms as transforms\n",
    "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix\n",
    "\n",
    "warnings.filterwarnings('ignore')\n",
    "DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
    "print(f\"Compute Device: {DEVICE}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Dataset Preparation (HAM10000 / ISIC)\n",
    "Using the 7 skin lesion diagnostic categories:\n",
    "`actinic keratosis`, `basal cell carcinoma`, `dermatofibroma`, `melanoma`, `nevus`, `pigmented benign keratosis`, `vascular lesion`."
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
    "dataset_path = kagglehub.dataset_download(\"nodoubttome/skin-cancer9-classesisic\")\n",
    "print(\"Dataset root:\", dataset_path)\n",
    "\n",
    "HAM_CLASSES = [\n",
    "    'actinic keratosis',\n",
    "    'basal cell carcinoma',\n",
    "    'dermatofibroma',\n",
    "    'melanoma',\n",
    "    'nevus',\n",
    "    'pigmented benign keratosis',\n",
    "    'vascular lesion'\n",
    "]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Edge Detection Implementations (Task 1)\n",
    "- **First-Order**: Sobel ($G_x, G_y, \\sqrt{G_x^2 + G_y^2}$), Prewitt\n",
    "- **Second-Order**: Laplacian ($\\nabla^2 I$), Laplacian of Gaussian (LoG)\n",
    "- **Multi-Stage**: Canny Edge Detector"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_sobel_edges(img_gray):\n",
    "    gx = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)\n",
    "    gy = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)\n",
    "    mag = cv2.magnitude(gx, gy)\n",
    "    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)\n",
    "\n",
    "def get_prewitt_edges(img_gray):\n",
    "    kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)\n",
    "    ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)\n",
    "    px = cv2.filter2D(img_gray, cv2.CV_32F, kx)\n",
    "    py = cv2.filter2D(img_gray, cv2.CV_32F, ky)\n",
    "    mag = np.sqrt(px**2 + py**2)\n",
    "    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)\n",
    "\n",
    "def get_laplacian_edges(img_gray):\n",
    "    lap = cv2.Laplacian(img_gray, cv2.CV_32F, ksize=3)\n",
    "    return cv2.normalize(np.abs(lap), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)\n",
    "\n",
    "def get_log_edges(img_gray, ksize=5, sigma=1.0):\n",
    "    blurred = cv2.GaussianBlur(img_gray, (ksize, ksize), sigma)\n",
    "    log = cv2.Laplacian(blurred, cv2.CV_32F, ksize=3)\n",
    "    return cv2.normalize(np.abs(log), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)\n",
    "\n",
    "def get_canny_edges(img_gray, low=50, high=150, ksize=3):\n",
    "    if ksize > 0:\n",
    "        blurred = cv2.GaussianBlur(img_gray, (ksize, ksize), 0)\n",
    "    else:\n",
    "        blurred = img_gray\n",
    "    return cv2.Canny(blurred, low, high)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Run Lab 03 Experiments Pipeline\n",
    "Executes Tasks 1 through 6, producing all tables and visualizations."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "!python run_lab3_experiments.py"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Visual Results and Comparative Analysis\n",
    "Display generated comparative figures."
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
    "print(\"1. Comparative Edge Detectors (Task 1):\")\n",
    "display(Image(\"figures/edge_detectors_comparison.png\"))\n",
    "\n",
    "print(\"2. Impact of Noise & Preprocessing (Task 2):\")\n",
    "display(Image(\"figures/noise_effect_comparison.png\"))\n",
    "\n",
    "print(\"3. Canny Parameter Analysis (Task 3):\")\n",
    "display(Image(\"figures/canny_parameter_analysis.png\"))\n",
    "\n",
    "print(\"4. Cross-Lab Confusion Matrices (Task 6):\")\n",
    "display(Image(\"figures/confusion_matrices_cross_lab.png\"))\n",
    "\n",
    "print(\"5. Cross-Lab Accuracy Comparison Bar Chart (Task 6):\")\n",
    "display(Image(\"figures/cross_lab_comparison_bar.png\"))"
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

with open("Lab3_Edge_Detection_Classification.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("Generated Lab3_Edge_Detection_Classification.ipynb successfully.")
