import os
import argparse
import numpy as np
import pandas as pd


def make_long_tail(df, num_classes, imb_ratio):

    img_max = len(df) / num_classes
    img_num_per_cls = []

    for cls_idx in range(num_classes):
        num = img_max * (imb_ratio ** (-cls_idx / (num_classes - 1)))
        img_num_per_cls.append(int(num))

    new_rows = []

    for cls, num in enumerate(img_num_per_cls):
        cls_rows = df[df.label == cls].sample(num)
        new_rows.append(cls_rows)

    return pd.concat(new_rows)


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--imbalance_ratio", type=int, default=100)
    ap.add_argument("--output_dir", required=True)

    args = ap.parse_args()

    train_df = pd.read_csv(os.path.join(args.dataset_root, "train.csv"))
    val_df = pd.read_csv(os.path.join(args.dataset_root, "val.csv"))
    test_df = pd.read_csv(os.path.join(args.dataset_root, "test.csv"))

    train_lt = make_long_tail(train_df, 100, args.imbalance_ratio)

    os.makedirs(args.output_dir, exist_ok=True)

    train_lt.to_csv(os.path.join(args.output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(args.output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(args.output_dir, "test.csv"), index=False)

    print("CIFAR100-LT generated.")


if __name__ == "__main__":
    main()