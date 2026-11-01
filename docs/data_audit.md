# Data audit (October 2026)

[日本語](data_audit.ja.md) · [Back to README](../README.md)

The dataset was audited with the duplicate-detection module of
[MedVision QA](https://github.com/MousJB/medvision-qa) (script `experiments/brain_mri/audit_mri_bt.py`,
results in `experiments/brain_mri/mri_bt_audit/`). Scope: the original Training (5,712) and
Testing (1,311) sets; the "Challenging" sets are excluded.

## Findings

| Finding | Value |
|---|---|
| Testing images with a byte-identical copy in Training | **216 / 1,311 (16.5%)** |
| — of which meningioma / no-tumor / pituitary | 106 / 105 / 5 |
| Identical duplicates with conflicting labels | 0 |
| Near-duplicates (perceptual hash, distance ≤ 4) with different labels | 50; the 24 inspected are all different images (false positives) |
| Images close to an image of the public figshare dataset (Cheng et al.) | 1,704 |

## What this shows

A sixth of the test set was already seen during training, so any evaluation on this
split is partly an evaluation on training data. The accuracies reported in the README
are therefore likely optimistic; the size of that bias has not been measured.

## What this does not show

An identical file proves an identical image, not an identical patient. The dataset has
no patient identifier, so it is impossible to check whether other slices of the same
patients are spread across Training and Testing; the real overlap may be larger. The
match with figshare is consistent with the dataset including figshare images, but does
not prove it.

## What a fair evaluation needs

A patient-level split. MedVision QA does this on the figshare dataset (which has
patient IDs): 64 held-out test patients, acceptance criteria fixed before training,
bootstrap confidence intervals per patient. A ResNet18 retrained there reached a
macro-F1 of 0.923 [0.867; 0.964]. It is a different model on a different task
(3 classes, no "no-tumor"), so this number is not directly comparable with the
results of this repository.
