import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torchvision import models

from fitz17k_data import build_eval_loader, class_names_from_mapping, default_class_to_idx
from model_factory import build_skin_model


def _set_deterministic(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def _build_bits_overrides(assignment_wts, assignment_acts):
    overrides = {}
    all_layers = set(assignment_wts.keys()) | set(assignment_acts.keys())
    for layer in sorted(all_layers):
        overrides[re.escape(layer)] = {
            "wts": int(assignment_wts.get(layer, 8)),
            "acts": int(assignment_acts.get(layer, 8)),
        }
    return overrides


def _load_checkpoint_payload(checkpoint_path):
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        meta = {k: v for k, v in checkpoint.items() if k != "model_state_dict"}
        return state_dict, meta
    return checkpoint, {}


def _safe_load_state_dict(model, state_dict):
    try:
        model.load_state_dict(state_dict)
        return
    except RuntimeError as original_error:
        pass

    if all(str(k).startswith("module.") for k in state_dict.keys()):
        stripped = {str(k)[7:]: v for k, v in state_dict.items()}
        try:
            model.load_state_dict(stripped)
            return
        except RuntimeError as stripped_error:
            raise RuntimeError(
                "Failed to load checkpoint state_dict after removing 'module.' prefix"
            ) from stripped_error

    raise RuntimeError("Failed to load checkpoint state_dict into model") from original_error


def _compute_binary_metrics(predictions, targets):
    tp = int(np.sum((predictions == 1) & (targets == 1)))
    tn = int(np.sum((predictions == 0) & (targets == 0)))
    fp = int(np.sum((predictions == 1) & (targets == 0)))
    fn = int(np.sum((predictions == 0) & (targets == 1)))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2.0 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0

    return {
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


class SkinDistillerBackend:
    def __init__(
        self,
        data_csv,
        image_dir,
        checkpoint,
        eval_batches,
        output_dir,
        seed,
        batch_size=32,
        eval_split="test",
        distiller_dir="/home/mussa/skin-fairness-ptq/distiller",
        label_space="auto",
        arch="resnet18",
    ):
        self.data_csv = str(Path(data_csv).resolve())
        self.image_dir = str(Path(image_dir).resolve())
        self.checkpoint = str(Path(checkpoint).resolve())
        self.eval_batches = int(eval_batches)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.eval_split = str(eval_split)
        self.distiller_dir = str(Path(distiller_dir).resolve())
        self.arch = str(arch)

        if self.distiller_dir not in sys.path:
            sys.path.insert(0, self.distiller_dir)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cache = {}
        self.eval_count = 0

        self.base_state_dict, ckpt_meta = _load_checkpoint_payload(self.checkpoint)
        self.label_space = str(label_space)
        if self.label_space == "auto":
            self.label_space = str(ckpt_meta.get("label_space", "binary"))

        ckpt_class_to_idx = ckpt_meta.get("class_to_idx")
        if ckpt_class_to_idx is None:
            self.class_to_idx = default_class_to_idx(self.label_space)
        else:
            self.class_to_idx = {str(k): int(v) for k, v in ckpt_class_to_idx.items()}

        self.class_names = class_names_from_mapping(self.class_to_idx)
        self.num_classes = len(self.class_names)

        self.eval_loader, split_meta = build_eval_loader(
            csv_path=self.data_csv,
            image_dir=self.image_dir,
            label_space=self.label_space,
            class_to_idx=self.class_to_idx,
            split=self.eval_split,
            batch_size=self.batch_size,
            seed=self.seed,
        )

        if int(split_meta["num_classes"]) != int(self.num_classes):
            raise ValueError(
                "Checkpoint class mapping does not match dataset encoding: "
                f"checkpoint={self.num_classes}, dataset={split_meta['num_classes']}"
            )

        self.eval_samples = int(split_meta["num_samples_eval"])

    def load_cache(self, cache_path):
        path = Path(cache_path)
        if path.exists():
            self.cache = json.loads(path.read_text())

    def save_cache(self, cache_path):
        Path(cache_path).write_text(json.dumps(self.cache, indent=2) + "\n")

    def _build_model(self):
        model = build_skin_model(arch=self.arch, num_classes=self.num_classes, pretrained=False)
        _safe_load_state_dict(model, self.base_state_dict)
        model.to(self.device)
        model.eval()
        return model

    def load_model_for_inventory(self):
        _set_deterministic(self.seed)
        return self._build_model()

    def evaluate_baseline(self):
        payload = {
            "mode": "baseline",
            "checkpoint": self.checkpoint,
            "eval_batches": self.eval_batches,
            "seed": self.seed,
            "split": self.eval_split,
            "label_space": self.label_space,
            "num_classes": self.num_classes,
            "class_to_idx": self.class_to_idx,
        }
        key = self._cache_key(payload)
        if key in self.cache:
            return self.cache[key]

        _set_deterministic(self.seed)
        model = self._build_model()
        metrics = self._evaluate_model(model)
        self.cache[key] = metrics
        self.eval_count += 1
        return metrics

    def evaluate_quantized(self, assignment_wts=None, assignment_acts=None, bits_acts=8, default_bits_wts=8):
        normalized_wts = {k: int(v) for k, v in sorted((assignment_wts or {}).items())}
        normalized_acts = {k: int(v) for k, v in sorted((assignment_acts or {}).items())}
        payload = {
            "mode": "quantized",
            "checkpoint": self.checkpoint,
            "eval_batches": self.eval_batches,
            "seed": self.seed,
            "split": self.eval_split,
            "label_space": self.label_space,
            "num_classes": self.num_classes,
            "class_to_idx": self.class_to_idx,
            "bits_acts": int(bits_acts),
            "default_bits_wts": int(default_bits_wts),
            "assignment_wts": normalized_wts,
            "assignment_acts": normalized_acts,
        }
        key = self._cache_key(payload)
        if key in self.cache:
            return self.cache[key]

        _set_deterministic(self.seed)
        model = self._build_model()

        from distiller.quantization import SymmetricLinearQuantizer

        bits_overrides = _build_bits_overrides(normalized_wts, normalized_acts)
        quantizer = SymmetricLinearQuantizer(
            model,
            bits_activations=int(bits_acts),
            bits_parameters=int(default_bits_wts),
            bits_overrides=bits_overrides,
        )
        quantizer.prepare_model()

        result = self._evaluate_model(model)
        result["assignment_wts"] = normalized_wts
        result["assignment_acts"] = normalized_acts
        self.cache[key] = result
        self.eval_count += 1
        return result

    def _evaluate_model(self, model):
        criterion = nn.CrossEntropyLoss()
        losses = []
        preds = []
        targets = []
        groups = []

        max_batches = self.eval_batches if self.eval_batches > 0 else None
        total = 0
        top1_correct = 0
        top5_correct = 0
        topk = min(5, self.num_classes)

        model.eval()
        with torch.no_grad():
            for batch_idx, (images, labels, fitz_batch) in enumerate(self.eval_loader, start=1):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = model(images)

                loss = criterion(outputs, labels)
                losses.append(loss.item() * images.size(0))

                batch_preds = torch.argmax(outputs, dim=1)
                top1_correct += int((batch_preds == labels).sum().item())
                preds.extend(batch_preds.cpu().numpy().tolist())
                targets.extend(labels.cpu().numpy().tolist())
                groups.extend(fitz_batch.cpu().numpy().tolist())

                if topk > 1:
                    topk_idx = torch.topk(outputs, k=topk, dim=1).indices
                    matches = topk_idx.eq(labels.unsqueeze(1)).any(dim=1)
                    top5_correct += int(matches.sum().item())
                else:
                    top5_correct += int((batch_preds == labels).sum().item())

                total += int(images.size(0))

                if max_batches is not None and batch_idx >= max_batches:
                    break

        n = max(1, int(total))
        top1 = 100.0 * float(top1_correct) / float(n)
        top5 = 100.0 * float(top5_correct) / float(n)
        avg_loss = float(sum(losses) / float(n))

        preds_np = np.array(preds, dtype=np.int64)
        targets_np = np.array(targets, dtype=np.int64)
        groups_np = np.array(groups, dtype=np.int64)

        overall_acc = float(top1_correct) / float(n)
        group_accuracies = {}
        for g in sorted(set(groups_np.tolist())):
            mask = groups_np == g
            g_total = int(mask.sum())
            g_correct = int((preds_np[mask] == targets_np[mask]).sum())
            group_accuracies[str(g)] = float(g_correct) / float(g_total) if g_total > 0 else 0.0
        unfairness_score = float(sum(abs(acc - overall_acc) for acc in group_accuracies.values()))

        labels_range = list(range(self.num_classes))
        macro_f1 = float(f1_score(targets_np, preds_np, average="macro", labels=labels_range, zero_division=0))
        weighted_f1 = float(f1_score(targets_np, preds_np, average="weighted", labels=labels_range, zero_division=0))

        result = {
            "top1": float(top1),
            "top5": float(top5),
            "loss": avg_loss,
            "accuracy": float(top1 / 100.0),
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "n_samples": int(n),
            "num_classes": int(self.num_classes),
            "label_space": self.label_space,
            "class_names": list(self.class_names),
            "eval_split": self.eval_split,
            "unfairness_score": unfairness_score,
            "group_accuracies": group_accuracies,
            "command": "inprocess:skin_distiller_backend",
        }

        if self.num_classes == 2:
            result.update(_compute_binary_metrics(preds_np, targets_np))

        return result

    @staticmethod
    def _cache_key(payload):
        s = json.dumps(payload, sort_keys=True)
        return hashlib.sha1(s.encode("utf-8")).hexdigest()
