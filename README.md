# skin-fairness-ptq

Mixed-precision post-training quantization (PTQ) with skin-tone fairness analysis on the Fitzpatrick17k dermatology dataset. Companion code for an IEEE Embedded Systems Letters submission from the EAS Lab at Eastern Michigan University.

## What this is

We ask: can sensitivity-aware mixed-precision PTQ reduce the unfairness gap that uniform quantization introduces on skin-tone subgroups, without sacrificing overall accuracy?

The repo evaluates six CNN architectures (VGG11, ResNet18/34/50, MobileNetV2, ShuffleNetV2 x1.0) on a 9-class Fitzpatrick17k task and compares three search strategies — **uniform baselines** (W8A8, W7A7, W6A6, ...), a **greedy** UD-constrained reference, and **NSGA-II** multi-objective search — across three objectives:

1. Top-1 accuracy
2. Spantidi unfairness *U_D* = Σ\_g |acc\_g − acc\_overall| over Fitzpatrick groups I–VI
3. Average bitwidth (proxy for compute / memory cost)

Energy is estimated post-hoc with Stanford's `nn_dataflow` for VGG/ResNet and an analytical MAC model for the depthwise-conv architectures.

## Repository layout

```
skin-fairness-ptq/
├── backend/        # PTQ + Fitz17k evaluation backend (Distiller-backed)
├── search/         # Greedy + uniform-sweep search drivers
├── scripts/        # NSGA-II driver, energy sim, aggregation, paper exports
├── results/        # Curated paper-ready CSVs + Excel workbooks
├── run.sh          # Bootstraps .venv, clones upstream Distiller, runs CIFAR sanity check
├── export_results.py
├── requirements.txt
├── CONTRIBUTING.md
└── LICENSE
```

See `CONTRIBUTING.md` for a per-file walkthrough.

## Dependencies and data

- **Python**: tested on 3.8+. Core deps in `requirements.txt`.
- **Upstream Distiller**: `run.sh` clones [`KeyKy/distiller`](https://github.com/KeyKy/distiller) into `./distiller/` and applies compatibility patches. Not in this repo.
- **Fitzpatrick17k**: Stanford-gated dataset. Apply for access via the [official source](https://github.com/mattgroh/fitzpatrick17k); we cannot redistribute images. Pipeline expects `fitzpatrick17k.csv` and the image tree at paths configured in `backend/fitz17k_data.py`.
- **Energy simulation** (optional): clone [`stanford-mast/nn_dataflow`](https://github.com/stanford-mast/nn_dataflow) and apply the source patches noted below.

## Quick start

```bash
git clone https://github.com/eas-lab-emich/skin-fairness-ptq.git
cd skin-fairness-ptq
chmod +x run.sh
./run.sh                  # creates .venv, clones upstream distiller, runs CIFAR PTQ sanity check
```

Then, with the Fitzpatrick17k data in place:

```bash
# 1. Train per-architecture FP32 baselines on Fitz17k
python backend/train_skin_fitz17k.py --arch resnet18

# 2. Uniform quantization sweep (baseline reference)
python search/run_uniform_sweep.py --arch resnet18

# 3. Greedy UD-constrained search
python search/greedy_search.py --arch resnet18

# 4. NSGA-II multi-objective search (pop=50, gen=30, ~1500 evals/arch)
python scripts/run_nsga2.py --arch resnet18

# 5. Aggregate Pareto results across archs
python scripts/aggregate_pareto.py
python scripts/build_pareto_excel.py

# 6. Energy estimation (requires nn_dataflow)
python scripts/energy_simulator.py
```

## Configuration

Several scripts contain default arguments that point at the original development host. **Before running, edit checkpoint and dataset paths** in:

- `backend/train_skin_fitz17k.py` — Fitz17k CSV + image dir
- `backend/fitz17k_data.py` — dataset locations
- `search/greedy_search.py`, `search/run_uniform_sweep.py` — checkpoint roots
- `scripts/run_nsga2.py`, `scripts/build_pareto_highlights.py`, `scripts/energy_simulator.py` — RESULTS / DATA paths

Or override via CLI flags where supported.

## Energy simulation setup

The `nn_dataflow` integration requires patches we developed for this work. Apply the following to your `nn_dataflow` checkout:

- `nn_dataflow/core/layer.py` — add `wbits, abits` fields to `ConvLayer` and `FCLayer`
- `nn_dataflow/core/scheduling.py` — scale `cost_op` by `(wbits * abits) / 64` in `_get_result`
- `nn_dataflow/core/pipeline_segment.py` — two sympy 1.13.1 compatibility fixes
- `nn_dataflow/nns/resnet50.py` — add `_ba_name()` and `build()`

Hardware config used in our experiments: `batch=16`, `array=12×14`, `nodes=1×1`, `regf=168 B`, `gbuf=55296 B`, `bus-width=64`, `--disable-interlayer-opt`.

MobileNetV2 and ShuffleNetV2 fall back to an analytical MAC model because `nn_dataflow` does not support depthwise convolutions; reported energy savings for those two are upper bounds (no memory / NoC cost).

## Results

Curated paper-facing artifacts live in `results/`:

- `results_summary.csv` — top-1 / U_D / avg bits for the six architectures across baseline, greedy, and NSGA-II
- `skin_fairness_ptq_results.xlsx` — per-architecture Pareto solutions and bit assignments
- `distiller_mixed_precision_results.xlsx` — earlier mixed-precision sweeps

Generated outputs (full Pareto fronts, per-iteration logs, energy workbooks) are written under `search/results/` and `scripts/logs/` and are gitignored.

## Citation

Manuscript in preparation for IEEE Embedded Systems Letters. Citation will be added upon publication.

## License

MIT — see `LICENSE`.
