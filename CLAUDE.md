# PTA — Prototype-Based Test-Time Adaptation

Research codebase for ICML 2026 paper: "Prototype-Based Test-Time Adaptation of Vision-Language Models".

---

## Running Experiments

### Cross-Domain Generalization

```bash
# ViT-B/16 backbone (all 10 CD datasets)
bash scripts/run_cd_benchmark_vit.sh

# ResNet-50 backbone
bash scripts/run_cd_benchmark_rn50.sh
```

### OOD Generalization (ImageNet variants)

```bash
bash scripts/run_ood_benchmark_vit.sh
bash scripts/run_ood_benchmark_rn50.sh
```

### Manual single-dataset run

```bash
python runner.py \
  --method pta \
  --config configs \
  --datasets caltech101 \
  --backbone ViT-B/16
```

`--datasets` accepts `/`-separated names (e.g. `I/A/V/R/S` or `caltech101/dtd`).  
`--wandb-log` is optional; omit to run offline.

### Running a different/new method

```bash
python runner.py \
  --method my_new_method \
  --config configs \
  --datasets caltech101 \
  --backbone ViT-B/16
```

Create `models/my_new_method.py` implementing `BaseAdapter` — no changes to `runner.py` needed.

---

## Key Files

```
runner.py             # Generic TTA entry point — use --method to select adapter
pta_runner.py         # DEPRECATED; kept for reference only
utils.py              # CLIP logit helpers, data loaders, config loader
configs/              # Per-dataset YAML configs (alpha and T params only)
scripts/              # Bash wrappers for benchmarks (ignore slurm_* files)
datasets/             # Dataset classes (ImageNet, Caltech101, etc.)
clip/                 # Local vendored copy of CLIP — do not replace with pip install
models/
  base.py             # BaseAdapter abstract class — implement this to add a new method
  pta.py              # PTA adapter (original paper method)
  exp1_*.py           # Experiment variants
  exp2_*.py
  exp3_*.py
  multi_proto_*.py
outputs/
  result.txt          # Results (appended, not overwritten)
```

---

## Algorithm Parameters

Hyperparameters live in `configs/<dataset>.yaml`. Only two params matter:

- `alpha` — weight of original text features vs. updated prototype (default: `0.01`)
- `T` — temperature for exponential update weight (default: `20.0`)

---

## Dataset Setup

Datasets are **not included**. Expected under `./data/` (override with `--data-root`).

Dataset name aliases for `--datasets`:
- `I` → ImageNet, `A` → ImageNet-A, `V` → ImageNetV2, `R` → ImageNet-R, `S` → ImageNet-Sketch
- Fine-grained datasets use full lowercase names: `caltech101`, `dtd`, `oxford_pets`, etc.

---

## Gotchas

- `clip/` is a **local vendored copy** of CLIP. Never `pip install clip` — `import clip` resolves to this local directory.
- `outputs/` must exist before running or the script crashes. Create it with `mkdir -p outputs`.
- `build_test_data_loader` in `utils.py` raises a bare string (`raise "Dataset is not..."`) — not a real exception in Python 3. Use `raise ValueError(...)` when extending dataset support.
- `pta_runner.py` is deprecated — all scripts use `runner.py`.
- `batch_size=1` is hardcoded in test loaders — TTA processes one sample at a time by design.
- Ignore all `scripts/slurm_*.sh` files — those are for a different cluster setup.
