# QaTS: Quantile-aware Temperature Scaling [ECCV 2026]

This repository contains the reference code for reproducing the core classification experiments of **QaTS**, a post-hoc calibration method based on quantile-aware temperature scaling.

This release focuses on the main evaluation setting used in the paper:
- **Uncalibrated model**
- **QaTS-calibrated model**

We provide scripts for:
- training image classifiers,
- fitting QaTS on a validation set,
- and evaluating calibration on a test set.

---

## Repository Structure

- `train_classifier.py` — trains a classifier from CSV annotations
- `run_qats.py` — evaluates the uncalibrated model and fits/evaluates QaTS
- `requirements.txt` — Python dependencies
- `README.md` — instructions for setup and reproduction

---

## Environment Setup

Create a Python environment using your preferred manager. Example using **conda**:

```bash
conda create -n qats python=3.10 -y
conda activate qats
pip install -r requirements.txt
```
## 📁  Datasets

Please follow [DATASETS.md](DATASETS.md) for instructions to prepare CIFAR-10, CIFAR-100, CIFAR-100-LT, and CIFAR-100-C.

## 🎯 Running QaTS

The evaluation pipeline consists of two steps:

1. Train a classifier on the target dataset.
2. Fit QaTS on the validation set and evaluate calibration on the test set.

---

## 1️⃣ Train a Classifier

The script `train_classifier.py` trains an image classifier and saves the best checkpoint.

Example: **CIFAR-100**

```bash
python train_classifier.py \
  --dataset cifar100 \
  --img_root ./datasets/cifar100 \
  --train_csv ./datasets/cifar100/train.csv \
  --val_csv ./datasets/cifar100/val.csv \
  --test_csv ./datasets/cifar100/test.csv \
  --save_dir ./checkpoints/cifar100 \
  --arch rn50 \
  --num_classes 100 \
  --epochs 120 \
  --batch 64 \
  --lr 0.01 \
  --wd 1e-4
```
This will produce:
```
checkpoints/
cifar100/
   cifar100_rn50_best.pth
```

Example: **CIFAR-10-LT(R-100)0**

```bash
python train_classifier.py \
  --dataset cifar100_lt_r100 \
  --img_root ./datasets/cifar100 \
  --train_csv ./datasets/cifar100_lt_r100/train.csv \
  --val_csv ./datasets/cifar100_lt_r100/val.csv \
  --test_csv ./datasets/cifar100_lt_r100/test.csv \
  --save_dir ./checkpoints/cifar100_lt_r100 \
  --arch rn50 \
  --num_classes 100 \
  --epochs 120 \
  --batch 64 \
  --lr 0.01 \
  --wd 1e-4
```
This will produce:
```
checkpoints/
cifar100/
   cifar100_rn50_best.pth
```

## 2️⃣ Run QaTS Calibration

After training the classifier, run QaTS using `run_qats.py`.

Example: **CIFAR-100**

```bash
python run_qats.py \
  --dataset cifar100 \
  --val_csv ./datasets/cifar100/val.csv \
  --test_csv ./datasets/cifar100/test.csv \
  --val_img_root ./datasets/cifar100 \
  --test_img_root ./datasets/cifar100 \
  --arch rn50 \
  --num_classes 100 \
  --ckpt ./checkpoints/cifar100/cifar100_rn50_best.pth \
  --batch 64
```


Example: **CIFAR-10-LT(R-100)0**

```bash
python run_qats.py \
  --dataset cifar100_lt_r100 \
  --val_csv ./datasets/cifar100_lt_r100/val.csv \
  --test_csv ./datasets/cifar100_lt_r100/test.csv \
  --val_img_root ./datasets/cifar100 \
  --test_img_root ./datasets/cifar100 \
  --arch rn50 \
  --num_classes 100 \
  --ckpt ./checkpoints/cifar100_lt_r100/cifar100_lt_r100_rn50_best.pth \
  --batch 64
```

