"""
export_bit_assignments.py
=========================
Exports per-layer bit assignments for every model and every configuration
into a single clean JSON file (bit_assignments.json).

Structure:
  {
    "resnet50": {
      "fp32":                   { "weights": {layer: bits}, "activations": {layer: bits} },
      "w8a8_uniform":           { ... },
      "w6a6_uniform":           { ... },
      "greedy_unconstrained":   { ... },
      "greedy_ud_constrained":  { ... }
    },
    ...
  }
"""

import json
from pathlib import Path

ROOT        = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "search" / "results"
OUT_PATH    = RESULTS_DIR / "bit_assignments.json"

ARCHS = ["vgg11", "resnet18", "resnet34", "resnet50", "mobilenet_v2", "shufflenet_v2_x1_0"]


def load(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def uniform_assignment(layer_names, bits):
    return {layer: bits for layer in layer_names}


def main():
    output = {}

    for arch in ARCHS:
        # Get layer names from the constrained run inventory
        inv_path = RESULTS_DIR / f"greedy_ud_constrained_{arch}_s42" / "layer_inventory.json"
        inv = load(inv_path)
        if not inv:
            print(f"  [SKIP] {arch} — no layer inventory found")
            continue
        layer_names = [l["name"] for l in inv["layers"]]

        arch_data = {}

        # FP32 — all layers at 32 bits
        arch_data["fp32"] = {
            "weights":     uniform_assignment(layer_names, 32),
            "activations": uniform_assignment(layer_names, 32),
        }

        # W8A8 uniform
        arch_data["w8a8_uniform"] = {
            "weights":     uniform_assignment(layer_names, 8),
            "activations": uniform_assignment(layer_names, 8),
        }

        # W6A6 uniform
        arch_data["w6a6_uniform"] = {
            "weights":     uniform_assignment(layer_names, 6),
            "activations": uniform_assignment(layer_names, 6),
        }

        # Greedy unconstrained (Phase 3)
        unc_dir = RESULTS_DIR / f"greedy_w6a6_{arch}_s42"
        unc_wt  = load(unc_dir / "assignment_weights.json")
        unc_act = load(unc_dir / "assignment_activations.json")
        if unc_wt and unc_act:
            arch_data["greedy_unconstrained"] = {
                "weights":     unc_wt,
                "activations": unc_act,
            }
        else:
            print(f"  [WARN] {arch} greedy_unconstrained assignments missing")

        # Greedy UD-constrained (Phase 4)
        con_dir = RESULTS_DIR / f"greedy_ud_constrained_{arch}_s42"
        con_wt  = load(con_dir / "assignment_weights.json")
        con_act = load(con_dir / "assignment_activations.json")
        if con_wt and con_act:
            arch_data["greedy_ud_constrained"] = {
                "weights":     con_wt,
                "activations": con_act,
            }
        else:
            print(f"  [WARN] {arch} greedy_ud_constrained assignments missing")

        output[arch] = arch_data
        print(f"  {arch}: {len(layer_names)} layers, {len(arch_data)} configs")

    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
