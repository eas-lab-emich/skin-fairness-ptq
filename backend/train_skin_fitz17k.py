import argparse
import csv
import json
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models

from fitz17k_data import build_train_val_test_loaders
from model_factory import build_skin_model, SUPPORTED_ARCHS


def set_deterministic(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def _build_model(num_classes, arch="resnet18", pretrained=True):
    return build_skin_model(arch=arch, num_classes=num_classes, pretrained=pretrained)


def _epoch_run(model, loader, criterion, device, num_classes, optimizer=None, max_batches=0):
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total = 0
    top1_correct = 0
    top5_correct = 0
    topk = min(5, int(num_classes))
    preds = []
    targets = []

    for batch_idx, (images, labels, _) in enumerate(loader, start=1):
        images = images.to(device)
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            outputs = model(images)
            loss = criterion(outputs, labels)
            if is_train:
                loss.backward()
                optimizer.step()

        batch_size = int(images.size(0))
        total_loss += float(loss.item()) * batch_size
        total += batch_size

        batch_preds = torch.argmax(outputs, dim=1)
        top1_correct += int((batch_preds == labels).sum().item())

        if topk > 1:
            topk_idx = torch.topk(outputs, k=topk, dim=1).indices
            top5_correct += int(topk_idx.eq(labels.unsqueeze(1)).any(dim=1).sum().item())
        else:
            top5_correct += int((batch_preds == labels).sum().item())

        preds.extend(batch_preds.detach().cpu().numpy().tolist())
        targets.extend(labels.detach().cpu().numpy().tolist())

        if max_batches > 0 and batch_idx >= max_batches:
            break

    n = max(1, total)
    labels_range = list(range(int(num_classes)))
    macro_f1 = float(f1_score(targets, preds, average="macro", labels=labels_range, zero_division=0))
    weighted_f1 = float(f1_score(targets, preds, average="weighted", labels=labels_range, zero_division=0))

    return {
        "loss": float(total_loss / n),
        "top1": float(100.0 * top1_correct / n),
        "top5": float(100.0 * top5_correct / n),
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "n_samples": int(total),
    }


def train(args):
    set_deterministic(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader, class_weights, meta = build_train_val_test_loaders(
        csv_path=args.skin_csv,
        image_dir=args.skin_image_dir,
        label_space=args.label_space,
        class_to_idx=None,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.workers,
        image_size=224,
    )

    model = _build_model(num_classes=meta["num_classes"], arch=args.arch, pretrained=not args.no_pretrained).to(device)
    class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"fitz17k_{args.label_space}_{args.arch}_best.pth"
    history_path = out_dir / f"fitz17k_{args.label_space}_{args.arch}_history.csv"
    summary_path = out_dir / f"fitz17k_{args.label_space}_{args.arch}_summary.json"

    best_val_loss = None
    best_val_top1 = None
    best_val_macro_f1 = None
    if args.select_metric == "val_loss":
        best_score = float("inf")
    else:
        best_score = -float("inf")
    best_epoch = 0
    history_rows = []

    print("=" * 80)
    print("FITZ17K MULTICLASS TRAINING")
    print("=" * 80)
    print(f"arch={args.arch} label_space={args.label_space} classes={meta['num_classes']} seed={args.seed}")
    print(f"split_sizes={meta['split_sizes']}")
    print(f"device={device}")

    for epoch in range(1, args.epochs + 1):
        train_metrics = _epoch_run(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            num_classes=meta["num_classes"],
            optimizer=optimizer,
            max_batches=args.max_train_batches,
        )
        val_metrics = _epoch_run(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=meta["num_classes"],
            optimizer=None,
            max_batches=args.max_eval_batches,
        )

        scheduler.step()
        lr_now = float(optimizer.param_groups[0]["lr"])

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_top1": train_metrics["top1"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_top1": val_metrics["top1"],
            "val_macro_f1": val_metrics["macro_f1"],
            "lr": lr_now,
        }
        history_rows.append(row)

        print(
            f"epoch={epoch:02d} "
            f"train_top1={train_metrics['top1']:.2f} "
            f"val_top1={val_metrics['top1']:.2f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"val_loss={val_metrics['loss']:.4f}"
        )

        if args.select_metric == "val_loss":
            current_score = float(val_metrics["loss"])
            is_better = current_score < best_score
        elif args.select_metric == "val_top1":
            current_score = float(val_metrics["top1"])
            is_better = current_score > best_score
        else:
            current_score = float(val_metrics["macro_f1"])
            is_better = current_score > best_score

        if is_better:
            best_score = current_score
            best_val_loss = float(val_metrics["loss"])
            best_val_top1 = float(val_metrics["top1"])
            best_val_macro_f1 = float(val_metrics["macro_f1"])
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "arch": args.arch,
                    "label_space": args.label_space,
                    "num_classes": int(meta["num_classes"]),
                    "class_names": list(meta["class_names"]),
                    "class_to_idx": dict(meta["class_to_idx"]),
                    "seed": int(args.seed),
                    "select_metric": args.select_metric,
                    "best_score": float(best_score),
                    "val_loss": float(val_metrics["loss"]),
                    "val_top1": float(val_metrics["top1"]),
                    "val_macro_f1": float(val_metrics["macro_f1"]),
                    "train_config": {
                        "epochs": int(args.epochs),
                        "batch_size": int(args.batch_size),
                        "lr": float(args.lr),
                        "workers": int(args.workers),
                        "max_train_batches": int(args.max_train_batches),
                        "max_eval_batches": int(args.max_eval_batches),
                        "skin_csv": str(Path(args.skin_csv).resolve()),
                        "skin_image_dir": str(Path(args.skin_image_dir).resolve()),
                    },
                    "saved_at": datetime.now().isoformat(timespec="seconds"),
                },
                ckpt_path,
            )
            print(f"saved_best={ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = _epoch_run(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        num_classes=meta["num_classes"],
        optimizer=None,
        max_batches=args.max_test_batches,
    )

    with history_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history_rows[0].keys()) if history_rows else ["epoch"])
        writer.writeheader()
        for row in history_rows:
            writer.writerow(row)

    summary = {
        "arch": args.arch,
        "label_space": args.label_space,
        "num_classes": int(meta["num_classes"]),
        "class_names": list(meta["class_names"]),
        "class_to_idx": dict(meta["class_to_idx"]),
        "seed": int(args.seed),
        "select_metric": args.select_metric,
        "best_score": float(best_score),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss if best_val_loss is not None else 0.0),
        "best_val_top1": float(best_val_top1 if best_val_top1 is not None else 0.0),
        "best_val_macro_f1": float(best_val_macro_f1 if best_val_macro_f1 is not None else 0.0),
        "test_top1": float(test_metrics["top1"]),
        "test_top5": float(test_metrics["top5"]),
        "test_macro_f1": float(test_metrics["macro_f1"]),
        "test_weighted_f1": float(test_metrics["weighted_f1"]),
        "checkpoint_path": str(ckpt_path.resolve()),
        "history_path": str(history_path.resolve()),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print("=" * 80)
    print("TRAINING COMPLETE")
    print(
        f"best_epoch={best_epoch} select_metric={args.select_metric} "
        f"best_score={best_score:.6f}"
    )
    print(
        f"test_top1={test_metrics['top1']:.2f} "
        f"test_top5={test_metrics['top5']:.2f} "
        f"test_macro_f1={test_metrics['macro_f1']:.4f}"
    )
    print(f"summary={summary_path}")
    print("=" * 80)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Fitz17k skin classifier (binary or 9-class)")
    parser.add_argument("--arch", default="resnet18", choices=SUPPORTED_ARCHS,
                        help="Model architecture to train.")
    parser.add_argument("--skin-csv", default="/home/mussa/skin_fairness_project/data/fitzpatrick17k.csv")
    parser.add_argument("--skin-image-dir", default="/home/mussa/fitzpatrick17k_data/data/finalfitz17k")
    parser.add_argument("--label-space", choices=["binary", "nine"], default="nine")
    parser.add_argument("--output-dir", default="/home/mussa/skin-fairness-ptq/backend/skin_checkpoints")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--max-test-batches", type=int, default=0)
    parser.add_argument("--select-metric", choices=["val_loss", "val_top1", "val_macro_f1"], default="val_top1")
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
