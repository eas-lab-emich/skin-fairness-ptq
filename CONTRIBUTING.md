# Contributing

Thanks for your interest in `skin-fairness-ptq`. This project is the public companion code for ongoing research at the EAS Lab (Eastern Michigan University) on mixed-precision quantization and demographic fairness. Contributions, issues, and questions are welcome.

## Code layout — what does what

### `backend/` — PTQ evaluation backend

- `distiller_backend.py` — subprocess wrapper around upstream Distiller's `compress_classifier.py`; the entry point all other code uses to evaluate a quantized configuration.
- `skin_distiller_backend.py` — subclass that adds Fitzpatrick group-stratified evaluation, worst-group accuracy (WGA), and Spantidi *U_D* unfairness scoring.
- `fitz17k_data.py` — dataset class and transforms for Fitzpatrick17k (9 disease classes × 6 skin-tone groups).
- `model_factory.py` — builds the six target architectures (VGG11, ResNet18/34/50, MobileNetV2, ShuffleNetV2 x1.0) with appropriate classifier heads.
- `layer_inventory.py` — enumerates `Conv2d` / `Linear` layers per architecture; exports per-layer metadata used by every search algorithm.
- `train_skin_fitz17k.py` — supervised training harness for FP32 baselines.
- `search.py` — generic grid / random-search driver kept around for ad-hoc experimentation.
- `tests/smoke_test.sh` — fast end-to-end sanity check.
- `tools/update_excel.py`, `tools/update_excel.sh` — incrementally appends new runs to the result workbook.
- `tools/validate_full_test.sh` — full-evaluation validator for a finished search run.

### `search/` — search strategies

- `run_uniform_sweep.py` — uniform-bitwidth baselines (W8A8, W7A7, W6A6, W5A5, W4A4) per architecture.
- `greedy_search.py` — greedy fairness-aware allocation; iteratively upgrades the layer with the best ΔWGA / Δmemory ratio. Used in the paper as a *reference point* — not the main optimizer.
- `tests/smoke_test.sh` — fast greedy sanity check.

### `scripts/` — orchestration, analysis, paper artifacts

- `run_nsga2.py` — **NSGA-II multi-objective search** (the main optimizer). Three objectives: −top-1, *U_D*, avg bits. pop=50, gen=30, seed=42.
- `recompute_pareto_spantidi.py` — recompute Pareto fronts using Spantidi *U_D* (sum of absolute deviations) instead of max−min UD.
- `regenerate_comparison_spantidi.py`, `regenerate_comparison_final_paper.py` — rebuild the cross-method comparison CSV.
- `aggregate_pareto.py` — consolidate UD-constrained greedy runs into `pareto_candidates.csv` + `final_comparison.csv`.
- `aggregate_results.py` — master Excel collation across runs.
- `build_pareto_excel.py` — per-architecture Pareto workbook (baselines, search paths, per-layer probes).
- `build_pareto_highlights.py` — pick three highlight solutions per arch (best top-1, best U_D, best avg bits) and filter by W8A8 domination.
- `export_bit_assignments.py` — flatten per-layer bit assignments into a single JSON for the paper.
- `energy_simulator.py` — `nn_dataflow`-based hardware energy estimation.
- `run_full_pipeline.sh`, `run_greedy_ud_constrained_all.sh`, `run_greedy_w6a6_all.sh`, `run_nsga2_queue.sh`, `run_nsga2_queue_final_paper.sh`, `launch_nsga2.sh` — shell orchestration that drives the per-architecture queues used to produce the paper numbers.

### Top level

- `run.sh` — bootstraps `.venv`, clones upstream Distiller, runs a CIFAR PTQ sanity check.
- `export_results.py` — small standalone Excel exporter for the headline paper numbers.
- `results/` — curated paper-ready artifacts (small enough to track in git).

## Development setup

```bash
git clone https://github.com/eas-lab-emich/skin-fairness-ptq.git
cd skin-fairness-ptq
./run.sh                              # bootstraps .venv and upstream Distiller
source .venv/bin/activate
pip install -r requirements.txt
```

For the Fitzpatrick17k pipeline you also need the dataset (Stanford-gated, see `README.md`) and edit hardcoded paths in the relevant scripts. For the energy pipeline you additionally need `nn_dataflow` with the patches noted in `README.md`.

## Code style

- Python 3.8+ syntax. Follow existing module conventions; we don't enforce a formatter, but match the surrounding file.
- Search algorithms read seeds from CLI flags (`--seed`); keep that convention so runs are reproducible.
- Heavy results (run directories, checkpoints, full Pareto JSONs) belong under gitignored paths — see `.gitignore`. Only small, curated artifacts go under `results/`.
- Don't redistribute Fitzpatrick17k images or any patient-identifiable data.

## Submitting changes

1. Fork the repo and create a feature branch off `main`.
2. Run the smoke tests: `bash backend/tests/smoke_test.sh` and `bash search/tests/smoke_test.sh`.
3. If your change touches search behavior, also re-run a small NSGA-II configuration (e.g., `--pop 10 --gen 2`) and verify the Pareto front shape is sensible.
4. Open a pull request describing the change, the experiments you ran, and any new dependencies.

## Reporting issues

Please file issues at <https://github.com/eas-lab-emich/skin-fairness-ptq/issues> with:

- The architecture and search method affected.
- The exact CLI invocation.
- The relevant slice of the log (full logs are large; trim to the failing step).
- Your environment (Python version, GPU, CUDA, OS).

## Contact

For research questions or collaboration inquiries, contact the EAS Lab at Eastern Michigan University.
