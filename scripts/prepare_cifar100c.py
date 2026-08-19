import os
import argparse
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Prepare CIFAR-100-C CSV annotations.")
    parser.add_argument("--cifar100c_root", required=True,
                        help="Root directory containing CIFAR-100-C npy files")
    parser.add_argument("--corruption", required=True,
                        help="Corruption type (e.g., gaussian_noise)")
    parser.add_argument("--severity", type=int, required=True,
                        help="Severity level (1-5)")
    parser.add_argument("--output_dir", required=True)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # CIFAR-100-C structure
    images_path = os.path.join(args.cifar100c_root, f"{args.corruption}.npy")
    labels_path = os.path.join(args.cifar100c_root, "labels.npy")

    if not os.path.exists(images_path):
        raise RuntimeError(f"Corruption file not found: {images_path}")

    images = np.load(images_path)
    labels = np.load(labels_path)

    # Each severity block contains 10k images
    start = (args.severity - 1) * 10000
    end = args.severity * 10000

    images = images[start:end]
    labels = labels[start:end]

    dataset_dir = os.path.join(args.output_dir, f"{args.corruption}_s{args.severity}")
    img_dir = os.path.join(dataset_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    rows = []

    for i in range(len(images)):
        img = images[i]
        label = int(labels[i])

        img_path = os.path.join(img_dir, f"{i}.png")

        from PIL import Image
        Image.fromarray(img).save(img_path)

        rows.append({
            "image_path": f"images/{i}.png",
            "label": label
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(dataset_dir, "test.csv"), index=False)

    print("CIFAR-100-C prepared")
    print(f"Saved to: {dataset_dir}")
    print(f"Total samples: {len(df)}")


if __name__ == "__main__":
    main()