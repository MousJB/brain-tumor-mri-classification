# Brain Tumor Classification from MRI Images using Deep Learning

This project explores different deep learning approaches for **brain tumor classification** from MRI scans using Python and PyTorch.

The goal is to classify MRI images into **4 classes**:
- Glioma
- Meningioma
- Pituitary
- No-tumor

> [!WARNING]
> **Data leakage found after publication (audit of October 2026).**
> **216 of the 1,311 Testing images (16.5%) are byte-identical copies of Training images**
> (meningioma 106, no-tumor 105, pituitary 5). The Training/Testing split of this dataset is
> therefore not independent, and **the accuracies reported below are likely optimistic**.
> The size of that bias has not been measured. See [Data audit](#-data-audit-october-2026).

---

## 📊 Dataset

**MRI Brain Tumor Dataset: 4-Class (7,023 Images)**  
Link: [Kaggle Dataset](https://www.kaggle.com/datasets/mohamadabouali1/mri-brain-tumor-dataset-4-class-7023-images)

- **Training set**: 5,712 images
- **Testing set**: Standard test images + 3 challenging datasets (Blurred, Noisy, Patient Motion Artifact)

**Class Distribution (Training):**
- No-tumor: 1,595 images
- Pituitary: 1,457 images
- Meningioma: 1,339 images
- Glioma: 1,321 images

All images are **224×224 pixels**.

---

## 🛠️ Technologies Used

- Python
- PyTorch + Torchvision
- OpenCV
- scikit-learn
- Matplotlib & Seaborn
- Google Colab

---

## 🚀 Models Implemented

### 1. SimpleCNN (Custom CNN from scratch)
- 3 convolutional blocks with Batch Normalization and Dropout
- Trained for 12 epochs
- Best validation accuracy: **~94.58%**

### 2. Fine-Tuning with Data Augmentation
- Added robust augmentations (Gaussian Blur, Gaussian Noise)
- Tested robustness on degraded datasets:
  - **Blurred Dataset**: **98.33%** accuracy
  - **Noisy Dataset**: **80.38%** accuracy
  - **Motion Artifact Dataset**: **93.92%** accuracy

### 3. Transfer Learning - ResNet18
- Pre-trained ResNet18 on ImageNet
- Only the final fully connected layer was trained (feature extraction)
- Enhanced augmentations including **random Motion Blur**
- Trained for 50 epochs

---

## 📈 Results Summary

| Model                    | Validation Accuracy | Blurred Acc | Noisy Acc | Motion Acc |
|--------------------------|---------------------|-------------|-----------|------------|
| SimpleCNN                | 94.58%             | -           | -         | -          |
| Fine-tuned SimpleCNN     | 98.09% (best)      | 98.33%      | 80.38%    | 93.92%     |
| ResNet18 (Transfer)      | Strong convergence | -           | -         | -          |

**How to read these numbers.** They were measured on the dataset's Training/Testing split,
which shares 216 identical images (see the audit below), so they should not be taken as
an estimate of performance on unseen patients. They are also **not reproducible from this
repository**: the notebook that produced them was not saved (the committed notebook is
empty), no model weights are versioned, and the origin of the "validation" set is not
documented. The Blurred, Noisy and Motion sets are synthetic degradations of the same
images: they are sensitivity tests, not real acquisition conditions.

---

## 🔍 Data audit (October 2026)

The dataset was audited with the duplicate-detection module of
[MedVision QA](https://github.com/MousJB/medvision-qa) (script `experiments/brain_mri/audit_mri_bt.py`,
results in `experiments/brain_mri/mri_bt_audit/`). Scope: the original Training (5,712) and
Testing (1,311) sets; the "Challenging" sets are excluded.

| Finding | Value |
|---|---|
| Testing images with a byte-identical copy in Training | **216 / 1,311 (16.5%)** |
| Identical duplicates with conflicting labels | 0 |
| Near-duplicates (perceptual hash, distance ≤ 4) with different labels | 50; the 24 inspected are all different images (false positives) |
| Images close to an image of the public figshare dataset (Cheng et al.) | 1,704 |

**What this shows.** A sixth of the test set was already seen during training, so any
evaluation on this split is partly an evaluation on training data.

**What this does not show.** An identical file proves an identical image, not an identical
patient. The dataset has no patient identifier, so it is impossible to check whether other
slices of the same patients are spread across Training and Testing; the real overlap may
be larger. The match with figshare is consistent with the dataset including figshare
images, but does not prove it.

**What a fair evaluation needs.** A patient-level split. MedVision QA does this on the
figshare dataset (which has patient IDs): 64 held-out test patients, acceptance criteria
fixed before training, bootstrap confidence intervals per patient. A ResNet18 retrained
there reached a macro-F1 of 0.923 [0.867; 0.964]. It is a different model on a different
task (3 classes, no "no-tumor"), so this number is not directly comparable with the table
above.

---

## 📁 Project Structure

brain-tumor-mri-classification/
├── notebooks/
│   └── Brain_Tumor_Classification.ipynb     # Main notebook
├── src/
│   ├── dataset.py
│   ├── models.py
│   └── utils.py
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md


---

## 🧪 How to Reproduce

1. Clone the repository:
   ```bash
   git clone https://github.com/MousJB/brain-tumor-mri-classification.git
   cd brain-tumor-mri-classification

2.Install dependencies
pip install -r requirements.txt

3.Download the dataset from Kaggle and place the Training/ and Testing/ folders inside a data/ directory.
4.Open Brain_Tumor_Classification.ipynb in Google Colab or Jupyter Notebook.



🎯 Project Objectives

Explore medical image classification with Deep Learning
Study the impact of data augmentation
Test model robustness on degraded images (blur, noise, motion artifacts)
Compare custom CNN vs Transfer Learning (ResNet18)


📌 Future Improvements

 Implement EfficientNet or Vision Transformers
 Add Grad-CAM for explainability
 k-fold cross-validation
 Model deployment with Gradio or Streamlit
 Re-evaluate on a deduplicated, patient-level split (the current Training/Testing split leaks, see Data audit)
