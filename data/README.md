# data/

Not versioned (see `.gitignore`). Download the
[MRI Brain Tumor Dataset: 4-Class (7,023 Images)](https://www.kaggle.com/datasets/mohamadabouali1/mri-brain-tumor-dataset-4-class-7023-images)
from Kaggle and unpack it here:

```
data/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    └── (same 4 sub-folders)
```

Class names are read from the sub-folders of `Training/`, so the exact folder
names do not matter as long as `Testing/` (and any extra test folder such as the
blurred, noisy or motion sets) uses the same ones.
