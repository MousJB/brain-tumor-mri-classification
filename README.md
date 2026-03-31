# Brain Tumor Classification from MRI Images using Deep Learning

This project explores different deep learning approaches for **brain tumor classification** from MRI scans using Python and PyTorch.

The goal is to classify MRI images into **4 classes**:
- Glioma
- Meningioma
- Pituitary
- No-tumor

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

---

## 📁 Project Structure

brain-tumor-mri-classification/
├── README.md
├── Brain_Tumor_Classification.ipynb
├── requirements.txt
├── models/
│   └── simple_cnn.pth                  # optional
├── src/
│   ├── dataset.py
│   ├── models.py
│   └── utils.py
└── data/                               # not included - download from Kaggle




---

## 🧪 How to Reproduce

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/brain-tumor-mri-classification.git
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
