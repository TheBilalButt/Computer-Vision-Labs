<div align="center">

# Computer Vision Labs.

<p align="center">
  <strong>Comprehensive Computer Vision Laboratory Portfolio</strong><br />
  BS Artificial Intelligence / Department of Computer Science<br />
  Author: <strong>Muhammad Bilal Butt</strong> / Roll Number: 42
</p>

<p align="center">
  <a href="https://github.com/TheBilalButt"><img src="https://img.shields.io/badge/AUTHOR-TheBilalButt-161b22?style=for-the-badge&logo=github&logoColor=white" alt="GitHub Profile" /></a>
  <a href="https://bilalbutt.tech/"><img src="https://img.shields.io/badge/PORTFOLIO-bilalbutt.tech-e50914?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Portfolio" /></a>
  <img src="https://img.shields.io/badge/CURRICULUM-LAB_01_TO_12-00c853?style=for-the-badge" alt="Curriculum" />
  <img src="https://img.shields.io/badge/STACK-OPENCV_/_PYTHON-007acc?style=for-the-badge&logo=python&logoColor=white" alt="Stack" />
</p>

</div>

<br />

## Overview.

This repository consolidates the complete set of academic laboratory assignments, research investigations, and practical implementations developed for the advanced Computer Vision and Machine Learning curriculum. 

It covers foundational spatial domain filtering, edge detection and boundary localization, local feature extraction (HOG and SIFT), evolutionary optimization with Genetic Algorithms for medical diagnosis, multimodal psychiatric visual state analysis, and end-to-end comparative machine learning regression model training.

<br />

## Laboratory Index.

| Lab Index | Focus and Topic | Primary Algorithms | Dataset / Domain | Interactive Link |
| :--- | :--- | :--- | :--- | :--- |
| **Lab 01** | OpenCV Fundamentals and Image Basics | Color Spaces (RGB, HSV, LAB), Geometric Warping, Histograms, Otsu Thresholding | Synthetic Patterns | [Launch Colab](https://colab.research.google.com/github/TheBilalButt/Computer-Vision-Labs/blob/main/Lab-01-OpenCV-Fundamentals/Lab01_OpenCV_Fundamentals.ipynb) |
| **Lab 02** | Image Filtering and Preprocessing | Gaussian, Median, Bilateral, Wiener, Mean Filtering, SVM Classifier | HAM10000 / ISIC Skin Lesions | [Launch Colab](https://colab.research.google.com/github/TheBilalButt/Computer-Vision-Labs/blob/main/Lab-02-Image-Filtering/Lab2_Skin_Lesion_Filtering.ipynb) |
| **Lab 03** | Edge Detection Techniques | Sobel, Prewitt, Laplacian of Gaussian, Canny, Quantitative Edge Benchmarking | ISIC Dermatoscopy | [Launch Colab](https://colab.research.google.com/github/TheBilalButt/Computer-Vision-Labs/blob/main/Lab-03-Edge-Detection/Lab3_Edge_Detection_Classification.ipynb) |
| **Lab 04** | Boundary Detection and Segmentation | Canny Edge Detection, Morphological Operations, Contour Analysis, IoU and Dice Score | HAM10000 Skin Lesions | [Launch Colab](https://colab.research.google.com/github/TheBilalButt/Computer-Vision-Labs/blob/main/Lab-04-Skin-Lesion-Boundary-Detection/Lab4_Skin_Lesion_Boundary_Detection.ipynb) |
| **Lab 05** | Industrial Defect Detection | Histogram of Oriented Gradients (HOG), Cell/Block Normalization, Linear SVM | Industrial Defect Images | [Launch Colab](https://colab.research.google.com/github/TheBilalButt/Computer-Vision-Labs/blob/main/Lab-05-HOG-Defect-Detection/Lab5_HOG_Industrial_Defect_Detection.ipynb) |
| **Lab 06** | Landmark Recognition and Localization | Scale-Invariant Feature Transform (SIFT), DoG Keypoints, Homography, Geometric Verification | Architectural Landmarks | [Launch Colab](https://colab.research.google.com/github/TheBilalButt/Computer-Vision-Labs/blob/main/Lab-06-SIFT-Landmark-Recognition/Lab6_SIFT_Landmark_Recognition.ipynb) |
| **Lab 07** | Evolutionary Medical Classification | Genetic Algorithm (GA) Feature Selection, HOG Descriptors, SVM Classification | PneumoniaMNIST Medical Images | [Launch Colab](https://colab.research.google.com/github/TheBilalButt/Computer-Vision-Labs/blob/main/Lab-07-GA-HOG-Medical-Classification/Lab07_GA_HOG_Medical_Classification.ipynb) |
| **Mid Exam** | Comparative Analysis: HOG vs Dense SIFT | HOG + SVM vs Dense SIFT + SVM Comparative Benchmark, ROC Curves, Confusion Matrices | BloodMNIST Cellular Dataset | [Launch Colab](https://colab.research.google.com/github/TheBilalButt/Computer-Vision-Labs/blob/main/Lab-Mid-HOG-vs-SIFT/42_BilalButt_LabMid.ipynb) |
| **Lab 08** | Feature Matching and Homography | SIFT, ORB, FLANN Matcher, Lowe Ratio Test, RANSAC Homography, Perspective Warping | Planar Scene Targets | [Launch Colab](https://colab.research.google.com/github/TheBilalButt/Computer-Vision-Labs/blob/main/Lab-08-Feature-Matching-Homography/Lab08_Feature_Matching_Homography.ipynb) |
| **Lab 09** | Multimodal Psychological State Analysis | Multimodal Feature Fusion, Facial Action Units, Vocal Formants, Linguistic Cues | Multimodal Affect Datasets | [View Full Paper](Lab-09-Psychological-State-Analysis/paper.pdf) |
| **Lab 10** | Model Training & Comparative ML Regression | 6 Regression Models (OLS, Decision Tree, Random Forest, XGBoost, KNN, MLP) | California Housing Dataset | [Launch Colab](https://colab.research.google.com/github/TheBilalButt/Computer-Vision-Labs/blob/main/Lab-10-Model-Training/Lab10_Model_Training.ipynb) |
| **Lab 11** | Student Week | None | None | [View Folder](Lab-11-Student-Week/) |
| **Lab 12** | CNN Feature Visualization & Parameter Audit | 5 Custom 3×3 Filters, ReLU, Max/Avg Pooling, Model A vs Model B Audit | High-Contrast Architectural Facade | [Launch Colab](https://colab.research.google.com/github/TheBilalButt/Computer-Vision-Labs/blob/main/Lab-12-CNN-Feature-Visualization/Lab12_CNN_Feature_Visualization.ipynb) |

<br />

## Detailed Laboratory Summaries.

### Lab 01: OpenCV Fundamentals and Image Processing Basics
- **Folder:** [`Lab-01-OpenCV-Fundamentals/`](Lab-01-OpenCV-Fundamentals/)
- Foundations of digital image representation, array manipulation via NumPy, conversions between RGB, Grayscale, HSV, and LAB spaces, geometric affine transformations, and intensity thresholding.

### Lab 02: Effect of Image Filtering on Skin-Lesion Classification
- **Folder:** [`Lab-02-Image-Filtering/`](Lab-02-Image-Filtering/)
- Quantitative assessment of noise suppression algorithms (Gaussian, Median, Mean, Bilateral) on dermoscopic skin lesion images from the HAM10000 dataset, measuring downstream classification impacts.

### Lab 03: Edge Detection Techniques and Classification Performance
- **Folder:** [`Lab-03-Edge-Detection/`](Lab-03-Edge-Detection/)
- Comparative evaluation of first order and second order derivative operators including Sobel, Prewitt, Robert's Cross, Laplacian of Gaussian, and Canny edge detectors.

### Lab 04: Skin Lesion Boundary Detection Using Canny Edge Detection
- **Folder:** [`Lab-04-Skin-Lesion-Boundary-Detection/`](Lab-04-Skin-Lesion-Boundary-Detection/)
- Lesion boundary extraction pipeline using optimal gradient thresholding, hysteresis tracking, morphological closing, and overlap validation against ground-truth segmentations.

### Lab 05: HOG-Based Industrial Defect Detection and Classification
- **Folder:** [`Lab-05-HOG-Defect-Detection/`](Lab-05-HOG-Defect-Detection/)
- End-to-end industrial surface inspection system using Histogram of Oriented Gradients (HOG) combined with Support Vector Machines. Includes a real-time inspection script and comprehensive experimental report.

### Lab 06: SIFT-Based Landmark Recognition and Geometric Localization
- **Folder:** [`Lab-06-SIFT-Landmark-Recognition/`](Lab-06-SIFT-Landmark-Recognition/)
- Multi-scale local invariant feature detection via Difference-of-Gaussians (DoG), descriptor generation, brute-force k-NN matching, and perspective transformation estimation. Includes a CLI recognition tool.

### Lab 07: Improving HOG Medical Classification Using Genetic Algorithms
- **Folder:** [`Lab-07-GA-HOG-Medical-Classification/`](Lab-07-GA-HOG-Medical-Classification/)
- Evolutionary optimization framework that uses Genetic Algorithms to select the most discriminative HOG feature bins on the PneumoniaMNIST biomedical dataset, maximizing cross-validated classification accuracy.

### Mid Examination: Comparative Analysis of HOG vs Dense SIFT on BloodMNIST
- **Folder:** [`Lab-Mid-HOG-vs-SIFT/`](Lab-Mid-HOG-vs-SIFT/)
- Empirical investigation comparing Histogram of Oriented Gradients against Dense Scale-Invariant Feature Transform descriptors for blood cell image classification on the BloodMNIST dataset. Includes full PDF submission report.

### Lab 08: Feature Matching and Homography Estimation
- **Folder:** [`Lab-08-Feature-Matching-Homography/`](Lab-08-Feature-Matching-Homography/)
- Advanced keypoint matching using FLANN indexing, ratio distance tests, robust outlier rejection using RANSAC, and homography-based perspective rectification.

### Lab 09: Psychological State Analysis Using Multimodal AI
- **Folder:** [`Lab-09-Psychological-State-Analysis/`](Lab-09-Psychological-State-Analysis/)
- Full academic research paper and software package evaluating multimodal behavioral indicators (facial affect, vocal tone, linguistic semantics) for psychological state estimation.

### Lab 10: Model Training and Comparative Machine Learning Regression
- **Folder:** [`Lab-10-Model-Training/`](Lab-10-Model-Training/)
- Empirical comparative benchmark of six regression architectures (Linear Regression, Decision Tree, Random Forest, XGBoost, K-Nearest Neighbors, MLP Neural Network) on the California Housing dataset. Evaluates leak-free data transformations, multi-metric performance rankings, and runtime complexity trade-offs with complete visualizations.

### Lab 11: Student Week
- **Folder:** [`Lab-11-Student-Week/`](Lab-11-Student-Week/)
- Student Week. No laboratory assignment required.

### Lab 12: CNN Feature Visualization, Parameter Audit & Critical Analysis
- **Folder:** [`Lab-12-CNN-Feature-Visualization/`](Lab-12-CNN-Feature-Visualization/)
- Visual inspection pipeline applying five custom 3×3 kernels (Sobel-X, Sobel-Y, Sharpen, Gaussian Blur, Emboss) across four sequential stages (Raw Conv, Post-ReLU, MaxPool, AvgPool), manual parameter count audit of Model A (3×3) vs Model B (5×5) verified with TensorFlow `model.summary()`, and critical analysis.

<br />

## Architecture and Structure.

```
Computer-Vision-Labs/
├── Lab-01-OpenCV-Fundamentals/
│   ├── Lab01_OpenCV_Fundamentals.ipynb
│   └── README.md
├── Lab-02-Image-Filtering/
│   ├── Lab2_Skin_Lesion_Filtering.ipynb
│   ├── run_experiments.py
│   └── README.md
├── Lab-03-Edge-Detection/
│   ├── Lab3_Edge_Detection_Classification.ipynb
│   ├── run_lab3_experiments.py
│   └── README.md
├── Lab-04-Skin-Lesion-Boundary-Detection/
│   ├── Lab4_Skin_Lesion_Boundary_Detection.ipynb
│   ├── run_lab4_experiments.py
│   └── README.md
├── Lab-05-HOG-Defect-Detection/
│   ├── Lab5_HOG_Industrial_Defect_Detection.ipynb
│   ├── realtime_inspector.py
│   ├── RESEARCH_PAPER.md
│   └── README.md
├── Lab-06-SIFT-Landmark-Recognition/
│   ├── Lab6_SIFT_Landmark_Recognition.ipynb
│   ├── landmark_recognizer_cli.py
│   └── README.md
├── Lab-07-GA-HOG-Medical-Classification/
│   ├── Lab07_GA_HOG_Medical_Classification.ipynb
│   ├── ga_hog_classification.py
│   └── README.md
├── Lab-Mid-HOG-vs-SIFT/
│   ├── 42_BilalButt_LabMid.ipynb
│   ├── 42_BilalButt_LabMid.pdf
│   └── README.md
├── Lab-08-Feature-Matching-Homography/
│   ├── Lab08_Feature_Matching_Homography.ipynb
│   └── README.md
├── Lab-09-Psychological-State-Analysis/
│   ├── paper.pdf
│   ├── paper.md
│   └── README.md
├── Lab-10-Model-Training/
│   ├── Lab10_Model_Training.ipynb
│   ├── Comparative_Machine_Learning_Regression.ipynb
│   ├── train_and_evaluate.py
│   ├── create_notebook.py
│   ├── requirements.txt
│   ├── README.md
│   └── visualizations/
│       ├── actual_vs_predicted_2x3.png
│       ├── rmse_comparison_bar.png
│       ├── r2_comparison_bar.png
│       └── model_comparison_dashboard.png
├── Lab-11-Student-Week/
│   └── README.md
├── Lab-12-CNN-Feature-Visualization/
│   ├── Lab12_CNN_Feature_Visualization.ipynb
│   ├── lab12_cnn_visualization.py
│   ├── README.md
│   └── visualizations/
│       ├── input_architecture.png
│       ├── part1_5x4_feature_grid.png
│       ├── part1_layman_relu_pooling_explainer.png
│       ├── part2_parameter_comparison_chart.png
│       └── part2_model_summaries.txt
└── README.md
```

<br />

## Setup and Environment.

### Prerequisites
- Python 3.9, 3.10, or 3.11
- Recommended: Virtual environment (`venv` or `conda`)

### Installation
```bash
git clone https://github.com/TheBilalButt/Computer-Vision-Labs.git
cd Computer-Vision-Labs
pip install -r Lab-05-HOG-Defect-Detection/requirements.txt
```

### Core Libraries
- OpenCV (`opencv-python`, `opencv-contrib-python`)
- NumPy and SciPy
- Scikit-Learn and Scikit-Image
- Matplotlib and Seaborn
- Streamlit and Pillow

<br />

## Author.

- **Name:** Muhammad Bilal Butt
- **GitHub:** [@TheBilalButt](https://github.com/TheBilalButt)
- **Website:** [bilalbutt.tech](https://bilalbutt.tech)
- **Institution:** BS Artificial Intelligence, Department of Computer Science
- **Instructor:** Dr. Jamal Hussain Shah

<br />

<div align="center">
  <sub>2026 MUHAMMAD BILAL BUTT. COMPUTER VISION LABORATORY REPOSITORY.</sub>
</div>
