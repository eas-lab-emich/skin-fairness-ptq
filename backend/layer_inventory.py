import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn


def load_model_for_inventory(arch, distiller_dir):
    import sys

    distiller_path = str(Path(distiller_dir).resolve())
    if distiller_path not in sys.path:
        sys.path.insert(0, distiller_path)

    from models import create_model

    dataset = "cifar10" if "cifar" in arch else "imagenet"
    model = create_model(False, dataset, arch, parallel=True)
    return model


def build_layer_inventory(model):
    layers = []
    total_weight_params = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if not hasattr(module, "weight") or module.weight is None:
                continue
            params = int(module.weight.numel())
            total_weight_params += params
            layers.append(
                {
                    "name": name,
                    "type": module.__class__.__name__,
                    "weight_params": params,
                    "quantizable": True,
                }
            )

    return {
        "num_layers": len(layers),
        "total_weight_params": total_weight_params,
        "layers": layers,
    }


def write_inventory(arch, distiller_dir, output_path):
    model = load_model_for_inventory(arch, distiller_dir)
    inv = build_layer_inventory(model)
    Path(output_path).write_text(json.dumps(inv, indent=2) + "\n")
    return inv


def main():
    parser = argparse.ArgumentParser(description="Export quantizable layer inventory")
    parser.add_argument("--arch", default="resnet56_cifar")
    parser.add_argument("--distiller-dir", default="/home/mussa/skin-fairness-ptq/distiller")
    parser.add_argument("--output", default="/home/mussa/skin-fairness-ptq/backend/results/layer_inventory.json")
    args = parser.parse_args()

    inv = write_inventory(args.arch, args.distiller_dir, args.output)
    print(f"Wrote inventory: {args.output}")
    print(f"Quantizable layers: {inv['num_layers']}, weight params: {inv['total_weight_params']}")


if __name__ == "__main__":
    main()
