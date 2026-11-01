# Brain Tumor Classification from MRI (PyTorch)

[日本語](README.ja.md)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![License](https://img.shields.io/badge/License-MIT-green)

Classification of brain MRI slices into **4 classes** (glioma, meningioma, pituitary
tumor, no tumor), comparing three approaches:

1. a **custom CNN** trained from scratch,
2. the same CNN trained with **degradation-style augmentation** (blur, noise, motion blur),
   then tested on blurred, noisy and motion-artifact versions of the images,
3. **ResNet18 transfer learning** (frozen ImageNet backbone, new classification head).

> [!WARNING]
> **The Training/Testing split of this dataset leaks.** An audit done after publication
> found that **216 of the 1,311 Testing images (16.5%) are byte-identical copies of
> Training images**. The accuracies below are therefore likely optimistic and should not
> be read as performance on unseen patients. Details: [docs/data_audit.md](docs/data_audit.md).

> This is a learning project, not a medical device.

---

## Dataset

[MRI Brain Tumor Dataset: 4-Class (7,023 Images)](https://www.kaggle.com/datasets/mohamadabouali1/mri-brain-tumor-dataset-4-class-7023-images) (Kaggle)

| Split | Images |
|---|---|
| Training | 5,712 (no-tumor 1,595 · pituitary 1,457 · meningioma 1,339 · glioma 1,321) |
| Testing | 1,311 + 3 "challenging" sets: Blurred, Noisy, Patient Motion Artifact |

Images are 224×224, read as grayscale. The challenging sets are synthetic degradations
of the same images: they measure sensitivity to degradation, not performance under
real acquisition conditions.

## Method

**Preprocessing** ([src/dataset.py](src/dataset.py)): grayscale → clipping to the
1st–99th percentile → z-score normalization on non-zero (brain) pixels.

**Augmentation** (optional, each with probability 0.5): Gaussian blur 5×5,
Gaussian noise (σ = 5), horizontal motion blur (kernel 3, 5 or 7).

**Models** ([src/models.py](src/models.py)):

| Model | Architecture | Trained parameters |
|---|---|---|
| SimpleCNN | 3 × (Conv 3×3 → BatchNorm → ReLU → MaxPool), FC 256, Dropout 0.5 | all |
| ResNet18 | ImageNet-pretrained, grayscale repeated to 3 channels | final FC layer only |

## Results

Measured in the original Google Colab notebook:

| Model | Validation acc. | Blurred | Noisy | Motion |
|---|---|---|---|---|
| SimpleCNN (12 epochs) | 94.58% | – | – | – |
| SimpleCNN + augmentation | **98.09%** | 98.33% | 80.38% | 93.92% |
| ResNet18, transfer (50 epochs) | not recorded | – | – | – |

Limits of these numbers:

- **Leak**: the Testing set shares 216 identical images with Training (see the warning above).
- **Not reproducible as-is**: the original notebook and weights were not saved, and the
  exact origin of the validation set was not documented. The scripts in this repository
  re-implement the pipeline from the saved modules, but have not yet been re-run to
  confirm these figures.
- **Image-level split**: the dataset has no patient IDs, so slices of one patient can sit
  on both sides of any split.

Among the degraded sets, Noisy is clearly the hardest (80.38%, against 93–98% for the others).

## Project structure

```
brain-tumor-mri-classification/
├── src/
│   ├── dataset.py        # Dataset class, preprocessing, augmentation
│   ├── models.py         # SimpleCNN, ResNet18 (transfer learning)
│   └── utils.py          # evaluation, confusion matrix, seeding
├── train.py              # training + validation split + checkpointing
├── evaluate.py           # accuracy and confusion matrix on test folders
├── data/                 # dataset goes here (not versioned, see data/README.md)
├── docs/
│   └── data_audit.md     # Training/Testing leak audit
├── requirements.txt
├── LICENSE
├── README.md
└── README.ja.md
```

`models/` (weights) and `results/` (metrics, figures) are created by the scripts and
ignored by git.

## Usage

```bash
git clone https://github.com/MousJB/brain-tumor-mri-classification.git
cd brain-tumor-mri-classification
pip install -r requirements.txt
```

Download the dataset into `data/` (layout in [data/README.md](data/README.md)), then:

```bash
# Custom CNN, with augmentation
python train.py --model simplecnn --epochs 12 --augment

# ResNet18 transfer learning
python train.py --model resnet18 --epochs 50 --augment

# Evaluation on the test set and the degraded sets
python evaluate.py --weights models/simplecnn.pth \
    --test-dirs data/Testing data/Blurred data/Noisy data/Motion
```

`train.py` holds out a stratified 15% of Training for validation (`--val-split`) and
keeps the best checkpoint on validation accuracy. `python train.py -h` lists all options.

## Future work

- [ ] Re-evaluate on a deduplicated, patient-level split (current split leaks)
- [ ] Grad-CAM for explainability
- [ ] EfficientNet / Vision Transformer
- [ ] k-fold cross-validation
- [ ] Demo with Gradio or Streamlit

## License

[MIT](LICENSE)
