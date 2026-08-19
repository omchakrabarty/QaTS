#!/usr/bin/env python3
import os
import random
import argparse
from typing import Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models
import timm


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------
class CSVDataset(Dataset):
    """
    Expected CSV columns:
      - label
      - image_id OR image_path

    If image_id is provided, the file is assumed to be:
        <img_root>/<image_id>.jpg

    If image_path is provided, it can be either absolute or relative to img_root.
    """

    def __init__(self, csv_path: str, img_root: str, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_root = img_root
        self.transform = transform

        if "label" not in self.df.columns:
            raise ValueError(f"{csv_path} must contain a 'label' column.")
        if ("image_id" not in self.df.columns) and ("image_path" not in self.df.columns):
            raise ValueError(f"{csv_path} must contain either 'image_id' or 'image_path'.")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        if "image_id" in self.df.columns:
            path = os.path.join(self.img_root, f"{row['image_id']}.jpg")
        else:
            rel_path = str(row["image_path"])
            path = rel_path if os.path.isabs(rel_path) else os.path.join(self.img_root, rel_path)

        label = int(row["label"])
        image = Image.open(path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# ------------------------------------------------------------
# Model zoo
# ------------------------------------------------------------
def build_model(arch: str, num_classes: int) -> nn.Module:
    arch = arch.lower()

    if arch == "rn18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if arch == "rn50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if arch == "dn121":
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model

    if arch == "dn201":
        model = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model

    if arch == "vits16":
        return timm.create_model(
            "vit_small_patch16_224",
            pretrained=True,
            num_classes=num_classes,
        )

    if arch == "vitb16":
        return timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,
            num_classes=num_classes,
        )

    raise ValueError("Unknown arch. Supported: rn18 | rn50 | dn121 | dn201 | vits16 | vitb16")


# ------------------------------------------------------------
# Class weights
# ------------------------------------------------------------
def compute_class_weights_from_csv(train_csv: str, num_classes: int) -> torch.Tensor:
    df = pd.read_csv(train_csv)
    counts = df["label"].value_counts().to_dict()

    weights = torch.zeros(num_classes, dtype=torch.float32)
    for c in range(num_classes):
        weights[c] = 1.0 / max(counts.get(c, 1), 1)

    weights = weights * (num_classes / weights.sum())  # normalize mean weight ~ 1
    return weights


# ------------------------------------------------------------
# Training / evaluation
# ------------------------------------------------------------
@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    criterion: nn.Module = None,
) -> Tuple[float, float]:
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0.0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)

        if criterion is None:
            loss = F.cross_entropy(logits, labels, reduction="sum")
        else:
            loss = criterion(logits, labels) * labels.numel()

        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.numel()
        total_loss += loss.item()

    acc = total_correct / total_samples
    avg_loss = total_loss / total_samples
    return acc, avg_loss


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    criterion: nn.Module,
) -> float:
    model.train()
    running_loss = 0.0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.numel()
        total_samples += labels.numel()

    return running_loss / total_samples


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train image classifier for QaTS evaluation.")
    parser.add_argument("--dataset", type=str, required=True, help="Used only for logging / checkpoint naming.")
    parser.add_argument("--img_root", type=str, required=True)
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)

    parser.add_argument(
        "--arch",
        type=str,
        default="rn50",
        choices=["rn18", "rn50", "dn121", "dn201", "vitb16", "vits16"],
    )
    parser.add_argument("--num_classes", type=int, required=True)

    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument(
        "--use_class_weights",
        action="store_true",
        help="Use inverse-frequency class weights in cross-entropy.",
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.0,
        help="Optional label smoothing, e.g. 0.05.",
    )

    parser.add_argument("--crop_scale_min", type=float, default=0.6)
    parser.add_argument("--rand_rot", type=float, default=10.0)

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------------- transforms ----------------
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(args.crop_scale_min, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(args.rand_rot),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # ---------------- datasets ----------------
    train_ds = CSVDataset(args.train_csv, args.img_root, transform=train_tf)
    val_ds = CSVDataset(args.val_csv, args.img_root, transform=eval_tf)
    test_ds = CSVDataset(args.test_csv, args.img_root, transform=eval_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ---------------- model ----------------
    model = build_model(args.arch, num_classes=args.num_classes).to(device)

    # ---------------- loss ----------------
    class_weights = None
    if args.use_class_weights:
        class_weights = compute_class_weights_from_csv(args.train_csv, args.num_classes).to(device)
        print("Using class weights:", class_weights.detach().cpu().numpy().round(3))

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing,
    )

    # ---------------- optimizer / scheduler ----------------
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.wd,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_ckpt_path = os.path.join(args.save_dir, f"{args.dataset}_{args.arch}_best.pth")
    last_ckpt_path = os.path.join(args.save_dir, f"{args.dataset}_{args.arch}_last.pth")

    best_val_acc = -1.0
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, criterion)
        scheduler.step()

        val_acc, val_nll = evaluate(model, val_loader, device)
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"{args.dataset} | {args.arch} | "
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"lr={lr_now:.6f} | "
            f"train_loss={train_loss:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"val_nll={val_nll:.4f}"
        )

        ckpt = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "best_val_acc": best_val_acc,
            "arch": args.arch,
            "dataset": args.dataset,
            "num_classes": args.num_classes,
            "args": vars(args),
        }
        torch.save(ckpt, last_ckpt_path)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            ckpt["best_val_acc"] = best_val_acc
            torch.save(ckpt, best_ckpt_path)
            print(f"  saved best checkpoint -> {best_ckpt_path}")

    # ---------------- final test with best checkpoint ----------------
    best_ckpt = torch.load(best_ckpt_path, map_location="cpu")
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.to(device)

    test_acc, test_nll = evaluate(model, test_loader, device)
    print(
        f"[BEST @ epoch {best_ckpt['epoch']}] "
        f"dataset={args.dataset} arch={args.arch} "
        f"test_acc={test_acc:.4f} test_nll={test_nll:.4f}"
    )


if __name__ == "__main__":
    main()