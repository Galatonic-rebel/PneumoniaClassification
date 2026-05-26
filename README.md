# PneumoniaClassification
A lightweight CNN classifier trained on chest X-ray images for binary pneumonia detection.

# Project Overview
This project implements an end-to-end deep learning pipeline for automated pneumonia detection from chest X-ray images. The model is designed to be lightweight and interpretable.

##  Dataset

**Source:** [Chest X-Ray Images (Pneumonia) — Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data)

| Split | Images |
|-------|--------|
| Train | 5,216  |
| Val   | 16     |
| Test  | 624    |

**Classes:** NORMAL vs PNEUMONIA (binary classification)

> Note: The dataset is imbalanced — pneumonia cases significantly outnumber normal cases. This was addressed using a weighted loss function.

---

##  Model Architecture

A simple 3-block CNN designed to balance performance and model size.

```
Input (128x128x3)
    ↓
Conv Block 1: Conv2d(3→16) → ReLU → MaxPool
    ↓
Conv Block 2: Conv2d(16→32) → ReLU → MaxPool
    ↓
Conv Block 3: Conv2d(32→64) → ReLU → AdaptiveAvgPool(4x4)
    ↓
FC: Linear(1024→64) → ReLU → Dropout(0.5)
    ↓
Output: Linear(64→2)
```

| Parameter     | Value       |
|---------------|-------------|
| Total params  | 89,314      |
| Input size    | 128×128     |
| Architecture  | Simple CNN  |

---

## Training Details

| Parameter   | Value                        |
|-------------|------------------------------|
| Epochs      | 15                           |
| Optimizer   | Adam (lr=0.001)              |
| Loss        | CrossEntropyLoss (weighted)  |
| Batch size  | 32                           |
| Best Val Acc| 0.8125 at epoch 9            |

**Data Augmentation (train only):**
- Random Horizontal Flip
- Random Rotation (±15°)
- Normalize (mean=0.5, std=0.5)

---

## Results

| Metric        | Value  |
|---------------|--------|
| Test Accuracy | 0.80   |
| ROC-AUC       | 0.92   |
| Test Loss     | 0.5634 |

**Classification Report:**

|           | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| NORMAL    | 0.90      | 0.51   | 0.65     | 234     |
| PNEUMONIA | 0.77      | 0.96   | 0.85     | 390     |
| Accuracy  |           |        | 0.80     | 624     |

> **Key clinical metric — PNEUMONIA recall of 0.96:** The model correctly identifies 96% of actual pneumonia cases, minimizing the risk of missed diagnoses.

---

## Grad-CAM Visualization

Grad-CAM (Gradient-weighted Class Activation Mapping) was used to visualize which regions of the X-ray the model focuses on when making predictions. 

---

##  How to Run

1. Clone the repository:
```bash
git clone https://github.com/yourusername/pneumonia-classifier.git
```

2. Open the notebook in Google Colab or Jupyter

3. Update the `base_path` variable to point to your dataset location

4. Run all cells top to bottom

---

##  Repository Structure

```
pneumonia-classifier/
├── pneumonia_classifier.ipynb   ← main notebook
├── README.md
└── images/
    ├── training_curves.png
    ├── confusion_matrix.png
    ├── roc_curve.png
    └── gradcam.png
```

---

## Dependencies

```
torch
torchvision
scikit-learn
matplotlib
seaborn
opencv-python
numpy
```

---

## Limitations

- Validation set is limited to 16 images due to the original dataset structure — validation metrics should be interpreted with caution.
- NORMAL recall of 0.51 indicates the model is biased toward predicting pneumonia, a known effect of class imbalance.
