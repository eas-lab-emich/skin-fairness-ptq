from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

BINARY_THREE_MAP = {
    "benign": 0,
    "non-neoplastic": 0,
    "malignant": 1,
}

NINE_CLASS_NAMES = [
    "benign dermal",
    "benign epidermal",
    "benign melanocyte",
    "genodermatoses",
    "inflammatory",
    "malignant cutaneous lymphoma",
    "malignant dermal",
    "malignant epidermal",
    "malignant melanoma",
]


def default_class_to_idx(label_space):
    if label_space == "binary":
        return {"non_malignant": 0, "malignant": 1}
    if label_space == "nine":
        return {name: idx for idx, name in enumerate(NINE_CLASS_NAMES)}
    raise ValueError(f"Unsupported label_space: {label_space}")


def class_names_from_mapping(class_to_idx):
    pairs = sorted(((int(v), str(k)) for k, v in class_to_idx.items()), key=lambda x: x[0])
    return [name for _, name in pairs]


def _normalize_class_to_idx(class_to_idx):
    return {str(k): int(v) for k, v in class_to_idx.items()}


def get_transforms(mode="eval", image_size=224):
    if mode == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
            ),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    return transforms.Compose([
        transforms.Resize(int(image_size * 256 / 224)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class Fitz17kDataset(Dataset):
    def __init__(self, csv_path, image_dir, label_space="binary", class_to_idx=None, transform=None):
        self.csv_path = str(Path(csv_path).resolve())
        self.image_dir = Path(image_dir).resolve()
        self.label_space = label_space
        self.transform = transform

        if class_to_idx is None:
            class_to_idx = default_class_to_idx(label_space)
        self.class_to_idx = _normalize_class_to_idx(class_to_idx)
        self.class_names = class_names_from_mapping(self.class_to_idx)
        self.num_classes = len(self.class_names)

        df = pd.read_csv(self.csv_path)
        df = df[df["fitzpatrick_scale"] != -1].copy()
        df["fitzpatrick_int"] = df["fitzpatrick_scale"].astype(int)

        if label_space == "binary":
            encoded = df["three_partition_label"].apply(lambda x: BINARY_THREE_MAP.get(x, np.nan))
            label_name = encoded.apply(lambda idx: self.class_names[int(idx)] if idx == idx else "")
        elif label_space == "nine":
            encoded = df["nine_partition_label"].apply(lambda x: self.class_to_idx.get(str(x), np.nan))
            label_name = df["nine_partition_label"].astype(str)
        else:
            raise ValueError(f"Unsupported label_space: {label_space}")

        df["label_idx"] = encoded
        df["label_name"] = label_name
        df = df.dropna(subset=["label_idx"])
        df["label_idx"] = df["label_idx"].astype(int)

        keep_rows = []
        paths = []
        for row_idx, row in df.iterrows():
            p = self.image_dir / f"{row['md5hash']}.jpg"
            if p.exists():
                keep_rows.append(row_idx)
                paths.append(p)

        self.df = df.loc[keep_rows].reset_index(drop=True)
        self.paths = paths

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(self.paths[idx]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = int(row["label_idx"])
        fitz = int(row["fitzpatrick_int"])
        return image, label, fitz

    def labels(self):
        return self.df["label_idx"].to_numpy(dtype=np.int64)

    def fitzpatrick(self):
        return self.df["fitzpatrick_int"].to_numpy(dtype=np.int64)


def _reduce_rare_strata(strata, labels, min_count=3):
    out = np.array(strata, dtype=object)
    counts = pd.Series(strata).value_counts()
    for group, count in counts.items():
        if int(count) < int(min_count):
            mask = out == group
            out[mask] = np.array([f"label_{int(x)}" for x in labels[mask]], dtype=object)
    return out


def create_split_indices(labels, fitzpatrick, random_state=42, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")

    indices = np.arange(len(labels))
    strata = np.array([f"{int(l)}_{int(f)}" for l, f in zip(labels, fitzpatrick)], dtype=object)
    strata = _reduce_rare_strata(strata, labels, min_count=3)

    try:
        train_idx, temp_idx = train_test_split(
            indices,
            test_size=(val_ratio + test_ratio),
            stratify=strata,
            random_state=random_state,
        )
    except ValueError:
        train_idx, temp_idx = train_test_split(
            indices,
            test_size=(val_ratio + test_ratio),
            stratify=labels,
            random_state=random_state,
        )

    relative_test_ratio = test_ratio / (val_ratio + test_ratio)
    temp_labels = labels[temp_idx]
    temp_fitz = fitzpatrick[temp_idx]
    temp_strata = np.array([f"{int(l)}_{int(f)}" for l, f in zip(temp_labels, temp_fitz)], dtype=object)
    temp_strata = _reduce_rare_strata(temp_strata, temp_labels, min_count=2)

    try:
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=relative_test_ratio,
            stratify=temp_strata,
            random_state=random_state,
        )
    except ValueError:
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=relative_test_ratio,
            stratify=temp_labels,
            random_state=random_state,
        )

    return list(train_idx), list(val_idx), list(test_idx)


def compute_class_weights(labels, num_classes):
    counts = np.bincount(labels, minlength=int(num_classes)).astype(np.float64)
    counts_safe = np.where(counts > 0, counts, 1.0)
    total = float(np.sum(counts))
    weights = total / (float(num_classes) * counts_safe)
    return torch.tensor(weights, dtype=torch.float32)


def build_eval_loader(
    csv_path,
    image_dir,
    label_space,
    class_to_idx,
    split,
    batch_size,
    seed,
    image_size=224,
):
    eval_ds = Fitz17kDataset(
        csv_path=csv_path,
        image_dir=image_dir,
        label_space=label_space,
        class_to_idx=class_to_idx,
        transform=get_transforms("eval", image_size=image_size),
    )

    labels = eval_ds.labels()
    fitz = eval_ds.fitzpatrick()
    train_idx, val_idx, test_idx = create_split_indices(labels, fitz, random_state=seed)
    eval_idx = val_idx if split == "val" else test_idx

    loader = DataLoader(
        Subset(eval_ds, eval_idx),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    meta = {
        "num_classes": eval_ds.num_classes,
        "class_names": eval_ds.class_names,
        "class_to_idx": eval_ds.class_to_idx,
        "num_samples_total": len(eval_ds),
        "num_samples_eval": len(eval_idx),
        "split": split,
        "split_sizes": {
            "train": len(train_idx),
            "val": len(val_idx),
            "test": len(test_idx),
        },
    }
    return loader, meta


def build_train_val_test_loaders(
    csv_path,
    image_dir,
    label_space,
    class_to_idx,
    batch_size,
    seed,
    num_workers=0,
    image_size=224,
):
    base_ds = Fitz17kDataset(
        csv_path=csv_path,
        image_dir=image_dir,
        label_space=label_space,
        class_to_idx=class_to_idx,
        transform=get_transforms("eval", image_size=image_size),
    )

    labels = base_ds.labels()
    fitz = base_ds.fitzpatrick()
    train_idx, val_idx, test_idx = create_split_indices(labels, fitz, random_state=seed)

    train_ds = Fitz17kDataset(
        csv_path=csv_path,
        image_dir=image_dir,
        label_space=label_space,
        class_to_idx=base_ds.class_to_idx,
        transform=get_transforms("train", image_size=image_size),
    )
    eval_ds = Fitz17kDataset(
        csv_path=csv_path,
        image_dir=image_dir,
        label_space=label_space,
        class_to_idx=base_ds.class_to_idx,
        transform=get_transforms("eval", image_size=image_size),
    )

    train_loader = DataLoader(
        Subset(train_ds, train_idx),
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        Subset(eval_ds, val_idx),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        Subset(eval_ds, test_idx),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
    )

    class_weights = compute_class_weights(labels[np.array(train_idx, dtype=np.int64)], base_ds.num_classes)
    meta = {
        "num_classes": base_ds.num_classes,
        "class_names": base_ds.class_names,
        "class_to_idx": base_ds.class_to_idx,
        "split_sizes": {
            "train": len(train_idx),
            "val": len(val_idx),
            "test": len(test_idx),
        },
        "num_samples_total": len(base_ds),
    }
    return train_loader, val_loader, test_loader, class_weights, meta
