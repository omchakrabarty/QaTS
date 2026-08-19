#!/usr/bin/env python3
import os
import json
import argparse
from typing import Optional, Tuple

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
# ECE
# ------------------------------------------------------------
class ECELoss(nn.Module):
    """Standard equal-width ECE in [0, 1]."""

    def __init__(self, n_bins: int = 15):
        super().__init__()
        boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = boundaries[:-1]
        self.bin_uppers = boundaries[1:]

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(probs, dim=1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for lower, upper in zip(self.bin_lowers, self.bin_uppers):
            in_bin = confidences.gt(lower.item()) * confidences.le(upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                acc_in_bin = accuracies[in_bin].float().mean()
                conf_in_bin = confidences[in_bin].mean()
                ece += torch.abs(conf_in_bin - acc_in_bin) * prop_in_bin
        return ece


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
@torch.no_grad()
def collect_logits(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    all_logits, all_labels = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())
    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


def compute_top1(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return preds.eq(labels).float().mean().item()


def conf_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    conf, _ = probs.max(dim=1)
    return conf


def quantile_from_sorted(conf: torch.Tensor, conf_sorted: torch.Tensor) -> torch.Tensor:
    ranks = torch.searchsorted(conf_sorted, conf, right=True)
    q = ranks.float() / conf_sorted.numel()
    return q


# ------------------------------------------------------------
# QaTS / ConCal
# ------------------------------------------------------------
class QaTS(nn.Module):
    """
    Instance-wise temperature:
        T(x) = clamp( a * (1 - q(x)) + b, Tmin, Tmax ), with a,b > 0
    where q(x) is the confidence quantile.
    """

    def __init__(self, Tmin: float = 0.5, Tmax: float = 5.0):
        super().__init__()
        self.Tmin = Tmin
        self.Tmax = Tmax
        self.raw_a = nn.Parameter(torch.tensor(0.0))
        self.raw_b = nn.Parameter(torch.tensor(0.0))

    def forward(self, logits: torch.Tensor, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        a = torch.exp(self.raw_a)
        b = torch.exp(self.raw_b)
        T = a * (1.0 - q) + b
        T = torch.clamp(T, min=self.Tmin, max=self.Tmax)
        scaled_logits = logits / T.unsqueeze(1)
        return scaled_logits, T


def fit_qats(
    logits_val: torch.Tensor,
    labels_val: torch.Tensor,
    max_iter: int = 200,
    lr: float = 0.05,
    weight_decay: float = 0.0,
    Tmin: float = 0.5,
    Tmax: float = 5.0,
) -> Tuple[QaTS, torch.Tensor]:
    device = logits_val.device
    scaler = QaTS(Tmin=Tmin, Tmax=Tmax).to(device)

    with torch.no_grad():
        conf_val = conf_from_logits(logits_val)
        conf_sorted = torch.sort(conf_val).values

    optimizer = torch.optim.AdamW(scaler.parameters(), lr=lr, weight_decay=weight_decay)

    best_nll = float("inf")
    best_state = None

    for _ in range(max_iter):
        optimizer.zero_grad()

        conf_now = conf_from_logits(logits_val)
        q_val = quantile_from_sorted(conf_now, conf_sorted)
        scaled_logits, _ = scaler(logits_val, q_val)

        loss = F.cross_entropy(scaled_logits, labels_val)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            scaled_logits_eval, _ = scaler(logits_val, q_val)
            nll = F.cross_entropy(scaled_logits_eval, labels_val).item()
            if nll < best_nll:
                best_nll = nll
                best_state = {k: v.detach().clone() for k, v in scaler.state_dict().items()}

    if best_state is not None:
        scaler.load_state_dict(best_state)

    return scaler, conf_sorted


# ------------------------------------------------------------
# Model zoo
# ------------------------------------------------------------
def build_cnn(arch: str, num_classes: int) -> nn.Module:
    arch = arch.lower()
    if arch == "rn18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if arch == "rn50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if arch == "dn121":
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model
    raise ValueError("Supported CNNs: rn18, rn50, dn121")


def infer_vit_family(state_dict: dict) -> str:
    keys = state_dict.keys()

    if ("class_token" in keys) or any(k.startswith("encoder.layers.") for k in keys) or any(
        k.startswith("conv_proj.") for k in keys
    ):
        return "torchvision"

    if ("cls_token" in keys) or any(k.startswith("blocks.") for k in keys) or any(
        k.startswith("patch_embed.") for k in keys
    ):
        return "timm"

    return "unknown"


def build_vit_from_ckpt(arch: str, num_classes: int, state_dict: dict) -> nn.Module:
    arch = arch.lower()
    family = infer_vit_family(state_dict)
    print(f"[ViT] detected checkpoint family: {family}")

    if family == "torchvision":
        if arch == "vitb16":
            if not hasattr(models, "vit_b_16"):
                raise RuntimeError("torchvision vit_b_16 unavailable, but checkpoint is torchvision-style.")
            model = models.vit_b_16(weights=None)
        elif arch == "vits16":
            if not hasattr(models, "vit_s_16"):
                raise RuntimeError("torchvision vit_s_16 unavailable, but checkpoint is torchvision-style.")
            model = models.vit_s_16(weights=None)
        else:
            raise ValueError("Supported ViTs: vitb16, vits16")
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        return model

    if family == "timm":
        name = "vit_base_patch16_224" if arch == "vitb16" else "vit_small_patch16_224"
        return timm.create_model(name, pretrained=False, num_classes=num_classes)

    raise RuntimeError("Could not infer ViT checkpoint family from state_dict.")


def build_pretrained_imagenet1k(arch: str, num_classes: int) -> nn.Module:
    """
    Build ImageNet-1K pretrained model with 1000-way classifier.
    """
    arch = arch.lower()
    if num_classes != 1000:
        raise ValueError("--pretrained requires --num_classes 1000.")

    if arch == "rn18":
        return models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    if arch == "rn50":
        return models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    if arch == "dn121":
        return models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)

    if arch in ["vitb16", "vits16"]:
        try:
            if arch == "vitb16" and hasattr(models, "vit_b_16") and hasattr(models, "ViT_B_16_Weights"):
                return models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
            if arch == "vits16" and hasattr(models, "vit_s_16") and hasattr(models, "ViT_S_16_Weights"):
                return models.vit_s_16(weights=models.ViT_S_16_Weights.IMAGENET1K_V1)
        except Exception as exc:
            print(f"[WARN] torchvision pretrained ViT failed ({arch}): {exc}")

        name = "vit_base_patch16_224" if arch == "vitb16" else "vit_small_patch16_224"
        return timm.create_model(name, pretrained=True, num_classes=1000)

    raise ValueError("Supported pretrained arch: rn18, rn50, dn121, vitb16, vits16")


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------
class CSVDataset(Dataset):
    """
    CSV must contain:
      - image_id or image_path
      - label (unless wnid2tv_json is provided)
    """

    def __init__(
        self,
        csv_path: str,
        img_root: str,
        transform=None,
        wnid2tv_json: Optional[str] = None,
    ):
        self.df = pd.read_csv(csv_path)
        self.img_root = img_root
        self.transform = transform

        if ("image_id" not in self.df.columns) and ("image_path" not in self.df.columns):
            raise ValueError(f"{csv_path} must contain 'image_id' or 'image_path'.")

        self.wnid2tv = None
        if wnid2tv_json is not None:
            with open(wnid2tv_json, "r") as f:
                self.wnid2tv = json.load(f)
            if len(self.wnid2tv) != 1000:
                raise ValueError("wnid2tv_json should map 1000 WNIDs to indices.")

        if self.wnid2tv is None and "label" not in self.df.columns:
            raise ValueError(f"{csv_path} must contain a 'label' column unless wnid2tv_json is provided.")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        if "image_id" in self.df.columns:
            rel_path = f"{row['image_id']}.jpg"
        else:
            rel_path = str(row["image_path"])

        img_path = rel_path if os.path.isabs(rel_path) else os.path.join(self.img_root, rel_path)

        if self.wnid2tv is not None:
            wnid = rel_path.split("/")[0]
            label = int(self.wnid2tv[wnid])
        else:
            label = int(row["label"])

        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


# ------------------------------------------------------------
# Printing
# ------------------------------------------------------------
def print_summary(
    uncal_acc: float,
    uncal_nll: float,
    uncal_ece: float,
    qats_acc: float,
    qats_nll: float,
    qats_ece: float,
) -> None:
    print("\n=== Calibration Summary ===")
    print(f"{'Method':<18} {'Acc(%)':>10} {'NLL':>10} {'ECE(%)':>10}")
    print("-" * 52)
    print(f"{'Uncalibrated':<18} {uncal_acc*100:>10.2f} {uncal_nll:>10.4f} {uncal_ece*100:>10.4f}")
    print(f"{'QaTS':<18} {qats_acc*100:>10.2f} {qats_nll:>10.4f} {qats_ece*100:>10.4f}")
    print("-" * 52)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate Uncalibrated and QaTS calibration.")
    parser.add_argument("--dataset", default="dataset", help="Dataset name for logging only.")

    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--val_img_root", required=True)
    parser.add_argument("--test_img_root", required=True)

    parser.add_argument("--arch", required=True, choices=["rn18", "rn50", "dn121", "vitb16", "vits16"])
    parser.add_argument("--num_classes", type=int, required=True)

    parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint path. Ignored if --pretrained is used.")
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet-1K pretrained model.")
    parser.add_argument(
        "--imagenet_wnid2tv",
        default=None,
        help="Optional JSON mapping WNID -> torchvision class index for ImageNet evaluation.",
    )

    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--ece_bins", type=int, default=15)
    parser.add_argument("--qats_iters", type=int, default=200)
    parser.add_argument("--qats_lr", type=float, default=0.05)
    parser.add_argument("--Tmin", type=float, default=0.5)
    parser.add_argument("--Tmax", type=float, default=5.0)

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_dataset = CSVDataset(
        csv_path=args.val_csv,
        img_root=args.val_img_root,
        transform=transform,
        wnid2tv_json=args.imagenet_wnid2tv,
    )
    test_dataset = CSVDataset(
        csv_path=args.test_csv,
        img_root=args.test_img_root,
        transform=transform,
        wnid2tv_json=args.imagenet_wnid2tv,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    if args.pretrained:
        model = build_pretrained_imagenet1k(args.arch, args.num_classes).to(device)
    else:
        if args.ckpt is None:
            raise ValueError("Either provide --ckpt or use --pretrained.")
        ckpt = torch.load(args.ckpt, map_location="cpu")
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

        if args.arch in ["vitb16", "vits16"]:
            model = build_vit_from_ckpt(args.arch, args.num_classes, state_dict)
        else:
            model = build_cnn(args.arch, args.num_classes)

        model.load_state_dict(state_dict, strict=True)
        model.to(device)

    logits_val, labels_val = collect_logits(model, val_loader, device)
    logits_test, labels_test = collect_logits(model, test_loader, device)

    logits_val = logits_val.to(device)
    labels_val = labels_val.to(device)
    logits_test = logits_test.to(device)
    labels_test = labels_test.to(device)

    ece_fn = ECELoss(n_bins=args.ece_bins)

    print(f"\n[Dataset={args.dataset}] arch={args.arch} classes={args.num_classes}")

    # ---------------- Uncalibrated ----------------
    uncal_acc = compute_top1(logits_test, labels_test)
    uncal_nll = F.cross_entropy(logits_test, labels_test).item()
    uncal_ece = ece_fn(logits_test, labels_test).item()

    print(
        f"Uncalibrated | "
        f"Acc: {uncal_acc*100:.2f} | "
        f"NLL: {uncal_nll:.4f} | "
        f"ECE: {uncal_ece*100:.4f}"
    )

    # ---------------- QaTS ----------------
    qats, conf_sorted = fit_qats(
        logits_val=logits_val,
        labels_val=labels_val,
        max_iter=args.qats_iters,
        lr=args.qats_lr,
        Tmin=args.Tmin,
        Tmax=args.Tmax,
    )

    with torch.no_grad():
        conf_test = conf_from_logits(logits_test)
        q_test = quantile_from_sorted(conf_test, conf_sorted)
        scaled_test_qats, temperatures = qats(logits_test, q_test)

        qats_acc = compute_top1(scaled_test_qats, labels_test)
        qats_nll = F.cross_entropy(scaled_test_qats, labels_test).item()
        qats_ece = ece_fn(scaled_test_qats, labels_test).item()

        a = torch.exp(qats.raw_a).item()
        b = torch.exp(qats.raw_b).item()

    print(
        f"QaTS | a: {a:.4f}, b: {b:.4f} | "
        f"T(min/mean/max)=({temperatures.min().item():.3f}/"
        f"{temperatures.mean().item():.3f}/"
        f"{temperatures.max().item():.3f})"
    )
    print(
        f"QaTS | "
        f"Acc: {qats_acc*100:.2f} | "
        f"NLL: {qats_nll:.4f} | "
        f"ECE: {qats_ece*100:.4f}"
    )

    print_summary(
        uncal_acc=uncal_acc,
        uncal_nll=uncal_nll,
        uncal_ece=uncal_ece,
        qats_acc=qats_acc,
        qats_nll=qats_nll,
        qats_ece=qats_ece,
    )


if __name__ == "__main__":
    main()