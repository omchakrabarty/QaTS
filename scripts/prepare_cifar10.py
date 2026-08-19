import os
import argparse
import pandas as pd
from torchvision.datasets import CIFAR10
from sklearn.model_selection import train_test_split


def save_split(dataset, out_dir, split):
    rows = []
    split_dir = os.path.join(out_dir, split)
    os.makedirs(split_dir, exist_ok=True)

    for i, (img, label) in enumerate(dataset):
        path = os.path.join(split_dir, f"{i}.png")
        img.save(path)

        rows.append({
            "image_path": f"{split}/{i}.png",
            "label": label
        })

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train_ds = CIFAR10(root="./datasets", train=True, download=True)
    test_ds = CIFAR10(root="./datasets", train=False, download=True)

    train_df = save_split(train_ds, args.output_dir, "train")
    test_df = save_split(test_ds, args.output_dir, "test")

    train_df, val_df = train_test_split(train_df, test_size=0.1, stratify=train_df["label"])

    train_df.to_csv(os.path.join(args.output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(args.output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(args.output_dir, "test.csv"), index=False)


if __name__ == "__main__":
    main()