"""
run_nsga2.py
============
NSGA-II multi-objective search for mixed-precision bit assignments.

Three objectives (all minimized):
  1. -top1          (maximize accuracy)
  2. ud_score       (minimize unfairness disparity)
  3. avg_bits       (minimize average bitwidth)

Each individual = per-layer bit assignments for weights + activations,
each value drawn from {6, 7, 8}.

Population seeded with existing greedy results + uniform baselines so we
don't start blind.

Usage:
  python3 scripts/run_nsga2.py --arch resnet50
  python3 scripts/run_nsga2.py --arch mobilenet_v2
"""

import argparse
import json
import logging
import sys
import os
import random
import numpy as np
from pathlib import Path
from datetime import datetime

# ── Project paths ─────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
BACKEND_DIR = ROOT / "backend"
RESULTS_DIR = ROOT / "search" / "results"
LOGS_DIR    = ROOT / "logs"
sys.path.insert(0, str(BACKEND_DIR))

from skin_distiller_backend import SkinDistillerBackend
from layer_inventory import build_layer_inventory

# ── pymoo ─────────────────────────────────────────────────────────────────────
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.core.population import Population
from pymoo.core.individual import Individual

# ── Per-arch config ───────────────────────────────────────────────────────────
ARCH_CONFIG = {
    "resnet50": {
        "checkpoint": "/home/mussa/skin-fairness-ptq/backend/skin_checkpoints/fitz17k_nine_resnet50_best.pth",
    },
    "mobilenet_v2": {
        "checkpoint": "/home/mussa/skin-fairness-ptq/backend/skin_checkpoints/fitz17k_nine_mobilenet_v2_best.pth",
    },
    "resnet18": {
        "checkpoint": "/home/mussa/skin-fairness-ptq/backend/skin_checkpoints/fitz17k_nine_resnet18_best.pth",
    },
    "resnet34": {
        "checkpoint": "/home/mussa/skin-fairness-ptq/backend/skin_checkpoints/fitz17k_nine_resnet34_best.pth",
    },
    "shufflenet_v2_x1_0": {
        "checkpoint": "/home/mussa/skin-fairness-ptq/backend/skin_checkpoints/fitz17k_nine_shufflenet_v2_x1_0_best.pth",
    },
    "vgg11": {
        "checkpoint": "/home/mussa/skin-fairness-ptq/backend/skin_checkpoints/fitz17k_nine_vgg11_best.pth",
    },
}

DATA_CSV   = "/home/mussa/skin_fairness_project/data/fitzpatrick17k.csv"
IMAGE_DIR  = "/home/mussa/fitzpatrick17k_data/data/finalfitz17k"
DISTILLER  = "/home/mussa/skin-fairness-ptq/distiller"
BIT_OPTS   = [6, 7, 8]


# ── Logging ───────────────────────────────────────────────────────────────────
def setup_logger(log_path):
    logger = logging.getLogger("nsga2")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fmt = logging.Formatter("%(asctime)s | %(message)s")
    fh  = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    sh  = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ── Encoding helpers ──────────────────────────────────────────────────────────
def encode(assignment_wts, assignment_acts, layers):
    """Dict assignments → flat int array [wt0, wt1, ..., act0, act1, ...]"""
    n = len(layers)
    x = np.zeros(2 * n, dtype=int)
    for i, l in enumerate(layers):
        x[i]     = assignment_wts.get(l, 6)
        x[n + i] = assignment_acts.get(l, 6)
    return x


def decode(x, layers):
    """Flat int array → (assignment_wts, assignment_acts) dicts"""
    n = len(layers)
    wts  = {l: int(x[i])     for i, l in enumerate(layers)}
    acts = {l: int(x[n + i]) for i, l in enumerate(layers)}
    return wts, acts


def avg_bits(x, layers, params_by_layer):
    n = len(layers)
    total_params = sum(params_by_layer.values())
    weighted = sum(params_by_layer[l] * int(x[i]) for i, l in enumerate(layers))
    return weighted / total_params


# ── Seed population ───────────────────────────────────────────────────────────
def build_seed_individuals(arch, layers, params_by_layer, pop_size, rng):
    seeds = []

    # Load existing greedy assignments
    for run_dir in [
        RESULTS_DIR / f"greedy_w6a6_{arch}_s42",
        RESULTS_DIR / f"greedy_ud_constrained_{arch}_s42",
    ]:
        wt_path  = run_dir / "assignment_weights.json"
        act_path = run_dir / "assignment_activations.json"
        if wt_path.exists() and act_path.exists():
            wts  = json.loads(wt_path.read_text())
            acts = json.loads(act_path.read_text())
            seeds.append(encode(wts, acts, layers))

    # W8A8
    seeds.append(encode({l: 8 for l in layers}, {l: 8 for l in layers}, layers))
    # W6A6
    seeds.append(encode({l: 6 for l in layers}, {l: 6 for l in layers}, layers))

    # Fill remaining with random individuals
    n_vars = 2 * len(layers)
    while len(seeds) < pop_size:
        x = np.array([rng.choice(BIT_OPTS) for _ in range(n_vars)], dtype=int)
        seeds.append(x)

    return np.array(seeds[:pop_size])


# ── pymoo problem ─────────────────────────────────────────────────────────────
class MixedPrecisionProblem(ElementwiseProblem):
    def __init__(self, backend, layers, params_by_layer, logger, cache, all_candidates, out_dir, save_every=50):
        n_vars = 2 * len(layers)
        super().__init__(
            n_var=n_vars,
            n_obj=3,
            xl=np.full(n_vars, 6),
            xu=np.full(n_vars, 8),
            vtype=int,
        )
        self.backend       = backend
        self.layers        = layers
        self.params        = params_by_layer
        self.logger        = logger
        self.cache         = cache          # dict: key → metrics
        self.all_candidates = all_candidates  # list of dicts (for saving)
        self.out_dir       = Path(out_dir)
        self.save_every    = save_every

    def _evaluate(self, x, out, *args, **kwargs):
        x = np.clip(np.round(x).astype(int), 6, 8)
        wts, acts = decode(x, self.layers)

        # Cache key
        key = json.dumps({"w": wts, "a": acts}, sort_keys=True)
        if key in self.cache:
            m = self.cache[key]
        else:
            m = self.backend.evaluate_quantized(
                assignment_wts=wts,
                assignment_acts=acts,
            )
            self.cache[key] = m

        top1     = m.get("top1", 0.0)
        ga       = m.get("group_accuracies", {})
        vals     = list(ga.values()) if ga else [0.0]
        # Spantidi U_D = sum_g |acc_g - acc_overall|; computed by backend.
        ud       = m.get("unfairness_score", 0.0)
        avg_b    = avg_bits(x, self.layers, self.params)
        macro_f1 = m.get("macro_f1", 0.0)
        wga      = min(vals)

        self.all_candidates.append({
            "top1": top1, "ud": ud, "avg_bits": avg_b,
            "wga": wga, "macro_f1": macro_f1,
            "assignment_wts": wts, "assignment_acts": acts,
        })

        n_evals = len(self.all_candidates)
        self.logger.info(
            "eval #%d | top1=%.2f  ud=%.4f  avg_bits=%.2f",
            n_evals, top1, ud, avg_b,
        )

        if n_evals % self.save_every == 0:
            try:
                self.backend.save_cache(self.out_dir / "eval_cache.json")
                with open(self.out_dir / "all_candidates.json", "w") as f:
                    json.dump(self.all_candidates, f, indent=2)
                self.logger.info("checkpoint saved (eval #%d)", n_evals)
            except Exception as e:
                self.logger.warning("checkpoint save failed: %s", e)

        out["F"] = [-top1, ud, avg_b]


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch",        required=True, choices=list(ARCH_CONFIG.keys()))
    parser.add_argument("--pop-size",    type=int, default=50)
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--out-suffix",  type=str, default="")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    cfg        = ARCH_CONFIG[args.arch]
    out_dir    = RESULTS_DIR / f"nsga2_{args.arch}_s{args.seed}{args.out_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path   = LOGS_DIR / f"nsga2_{args.arch}{args.out_suffix}.log"
    logger     = setup_logger(log_path)

    logger.info("=== NSGA-II  arch=%s  pop=%d  gen=%d ===",
                args.arch, args.pop_size, args.generations)

    # ── Backend ───────────────────────────────────────────────────────────────
    backend = SkinDistillerBackend(
        data_csv=DATA_CSV,
        image_dir=IMAGE_DIR,
        checkpoint=cfg["checkpoint"],
        eval_batches=0,
        output_dir=out_dir,
        seed=args.seed,
        batch_size=32,
        eval_split="test",
        distiller_dir=str(DISTILLER),
        label_space="nine",
        arch=args.arch,
    )
    cache_path = out_dir / "eval_cache.json"
    backend.load_cache(cache_path)

    # ── Layer inventory ───────────────────────────────────────────────────────
    model = backend.load_model_for_inventory()
    inventory = build_layer_inventory(model)
    layers = [e["name"] for e in inventory["layers"]]
    params_by_layer = {e["name"]: int(e["weight_params"]) for e in inventory["layers"]}
    logger.info("Layers: %d", len(layers))

    # ── Seed population ───────────────────────────────────────────────────────
    seed_X = build_seed_individuals(args.arch, layers, params_by_layer, args.pop_size, rng)
    logger.info("Seed population: %d individuals (%d from existing results)",
                len(seed_X), min(4, len(seed_X)))

    # ── Problem + algorithm ───────────────────────────────────────────────────
    cache          = {}
    all_candidates = []

    problem = MixedPrecisionProblem(backend, layers, params_by_layer, logger, cache, all_candidates, out_dir)

    sampling = seed_X  # pass directly as initial population

    algorithm = NSGA2(
        pop_size=args.pop_size,
        sampling=sampling,
        crossover=SBX(prob=0.9, eta=15, vtype=float, repair=None),
        mutation=PM(prob=1.0 / (2 * len(layers)), eta=20, vtype=float, repair=None),
        eliminate_duplicates=True,
    )

    termination = get_termination("n_gen", args.generations)

    logger.info("Starting NSGA-II (%d generations × %d pop = ~%d evals)...",
                args.generations, args.pop_size, args.generations * args.pop_size)

    result = minimize(
        problem,
        algorithm,
        termination,
        seed=args.seed,
        verbose=False,
    )

    # ── Save eval cache ───────────────────────────────────────────────────────
    backend.save_cache(cache_path)

    # ── Extract Pareto front ──────────────────────────────────────────────────
    pareto = []
    for i in range(len(result.X)):
        x        = np.clip(np.round(result.X[i]).astype(int), 6, 8)
        wts, acts = decode(x, layers)
        f        = result.F[i]
        pareto.append({
            "top1":           -float(f[0]),
            "ud":             float(f[1]),
            "avg_bits":       float(f[2]),
            "assignment_wts":  wts,
            "assignment_acts": acts,
        })

    pareto.sort(key=lambda r: r["top1"], reverse=True)

    # ── Save outputs ──────────────────────────────────────────────────────────
    all_path    = out_dir / "all_candidates.json"
    pareto_path = out_dir / "pareto_front.json"

    with open(all_path, "w") as f:
        json.dump(all_candidates, f, indent=2)
    with open(pareto_path, "w") as f:
        json.dump(pareto, f, indent=2)

    logger.info("Done. Total evals: %d", len(all_candidates))
    logger.info("Pareto front size: %d", len(pareto))
    logger.info("All candidates: %s", all_path)
    logger.info("Pareto front:   %s", pareto_path)

    # Quick summary
    logger.info("--- Pareto front (top1 sorted) ---")
    for r in pareto[:10]:
        logger.info("  top1=%.2f  ud=%.4f  avg_bits=%.2f", r["top1"], r["ud"], r["avg_bits"])


if __name__ == "__main__":
    main()
