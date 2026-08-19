# Dataset Preparation

This document describes how to prepare the datasets used in the QaTS experiments.

The code in this repository expects datasets to be provided through **CSV annotation files** specifying image paths and labels.

Supported datasets in this release:

- CIFAR-10
- CIFAR-100
- CIFAR-100-LT (R-10 / R-100)
- CIFAR-100-C

---

# Expected Dataset Format

Each dataset must contain CSV files with the following structure.

Example CSV:
```
image_path,label
train/airplane/img_0001.png,0
train/dog/img_0002.png,5
```
Required columns:
```
| Column | Description |
|------|-------------|
| `image_path` | relative or absolute path to the image |
| `label` | integer class label |
```
Images should be located inside a root directory specified by `--img_root`.

Example directory layout:
```
datasets/
cifar100/
train/
test/
train.csv
val.csv
test.csv
```

---

# 1. CIFAR-10

Download CIFAR-10 using torchvision:
```
python - <<EOF
import torchvision.datasets as d

d.CIFAR10(root=”./datasets”, train=True, download=True)
d.CIFAR10(root=”./datasets”, train=False, download=True)
EOF
```

Convert CIFAR-10 to an image dataset and generate CSV files:
```
python scripts/prepare_cifar10.py 
–output_dir ./datasets/cifar10

```

This will create:
```
datasets/cifar10/
train/
test/
train.csv
val.csv
test.csv

```
The validation set is typically created by splitting **10% of the training set**.

---

# 2. CIFAR-100

Download CIFAR-100:
```
python - <<EOF
import torchvision.datasets as d

d.CIFAR100(root=”./datasets”, train=True, download=True)
d.CIFAR100(root=”./datasets”, train=False, download=True)
EOF
```
Convert to image format and generate CSV files:
```
python scripts/prepare_cifar100.py 
–output_dir ./datasets/cifar100
```

This will create:
```
datasets/cifar100/
train/
test/
train.csv
val.csv
test.csv
```
---

# 3. CIFAR-100-LT (Long-Tailed)

CIFAR-100-LT is created by sampling the CIFAR-100 training set with a long-tailed class distribution.

Two imbalance ratios are used:

| Dataset | Imbalance Ratio |
|-------|----------------|
| CIFAR-100-LT (R-10) | mild imbalance |
| CIFAR-100-LT (R-100) | extreme imbalance |

Generate the datasets using:
```
python scripts/prepare_cifar100_lt.py 
–dataset_root ./datasets/cifar100 
–imbalance_ratio 10 
–output_dir ./datasets/cifar100_lt_r10

python scripts/prepare_cifar100_lt.py 
–dataset_root ./datasets/cifar100 
–imbalance_ratio 100 
–output_dir ./datasets/cifar100_lt_r100
```
Output structure:
```
datasets/
cifar100_lt_r10/
train.csv
val.csv
test.csv

cifar100_lt_r100/
train.csv
val.csv
test.csv
```
Note:

- The **test set remains the same as CIFAR-100**
- Only the **training distribution is imbalanced**

---

# 4. CIFAR-100-C (Corrupted CIFAR-100)

CIFAR-100-C evaluates calibration under distribution shift.

Download the dataset:
```
wget https://zenodo.org/record/3555552/files/CIFAR-100-C.tar
tar -xvf CIFAR-100-C.tar
```
The dataset contains **15 corruption types × 5 severity levels**.

Example corruptions:

- gaussian\_noise
- motion\_blur
- fog
- snow
- brightness
- jpeg\_compression

To evaluate a specific corruption:
```
python scripts/prepare_cifar100c.py \
  --cifar100c_root ./CIFAR-100-C \
  --corruption gaussian_noise \
  --severity 3 \
  --output_dir ./datasets/cifar100c
```
Example structure:
```
datasets/cifar100c/
gaussian_noise_s3/
test.csv
```
QaTS can then be evaluated using `run_qats.py`.

---

# Notes

- CIFAR images are originally **32×32**, but the training pipeline resizes them to **224×224** for compatibility.
- The CSV format allows easy extension to additional datasets.