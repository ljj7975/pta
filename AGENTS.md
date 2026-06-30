# AGENTS.md — PTA (Prototype-Based Test-Time Adaptation)

Research codebase for ICML 2026 paper: "Prototype-Based Test-Time Adaptation of Vision-Language Models".
Two independent sub-projects with different envs: image recognition (`pta`) and point cloud robustness (`pta_point`).

---

## Two Separate Environments

This repo has **two separate conda envs** — they are not interchangeable:

| Sub-project | Env | Python | PyTorch | Entry point |
|---|---|---|---|---|
| Image recognition | `pta` | 3.9 | 2.0.1 + CUDA 11.7 | `runner.py` |
| Point cloud | `pta_point` | 3.8.16 | 1.12.0 + CUDA 11.6 | `PTA_point/run_pta.py` |

The point cloud env requires building `dassl` in-place:
```bash
cd PTA_point/dassl/
python setup.py develop
```

---

## Running Experiments

### Image Recognition (cross-domain generalization)
```bash
# ViT-B/16 backbone — runs all 10 CD datasets in one shot
bash scripts/run_cd_benchmark_vit.sh

# ResNet-50 backbone
bash scripts/run_cd_benchmark_rn50.sh
```

### Image Recognition (OOD generalization — ImageNet variants)
```bash
# Datasets: I=ImageNet, V=ImageNetV2, R=ImageNet-R, S=ImageNet-Sketch, A=ImageNet-A
bash scripts/run_ood_benchmark_vit.sh
bash scripts/run_ood_benchmark_rn50.sh
```

### Point Cloud Robustness (ModelNet-C / SONN-C)
```bash
# Args: <gpu> <lm3d_model> <ckpt_path> <dataset> <sonn_variant> <npoints> <os_version> <ulip_version> <s2r_type>
# Loops over 7 corruption types automatically
bash ./PTA_point/scripts/eval_pta.sh 0 ulip weights/ulip/pointbert_ulip1.pt modelnet_c obj_only 1024 vitg14 ulip1 so_obj_only_9
bash ./PTA_point/scripts/eval_pta.sh 0 ulip weights/ulip/pointbert_ulip1.pt sonn_c obj_only 1024 vitg14 ulip1 so_obj_only_9
```
`eval_pta.sh` automatically iterates over: `add_global_2 add_local_2 dropout_global_2 dropout_local_2 jitter_2 rotate_2 scale_2`.  
To change corruption type or severity, edit the `cor_types` array inside `eval_pta.sh`.

### Manual single-dataset run
```bash
python runner.py \
  --method pta \
  --config configs \
  --datasets caltech101 \
  --backbone ViT-B/16
# --wandb-log is optional; omit to skip wandb
```
`--datasets` accepts `/`-separated names (e.g. `I/A/V/R/S` or `caltech101/dtd`).

### Running a different method
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
pta_runner.py         # DEPRECATED; kept for reference only; all scripts now use runner.py
utils.py              # CLIP logit helpers, data loaders, config loader
configs/              # Per-dataset YAML configs (only two params: alpha, T)
scripts/              # Bash + Slurm wrappers for benchmarks (all use runner.py --method pta)
datasets/             # Dataset classes (ImageNet, Caltech101, etc.)
clip/                 # Local copy of CLIP — do not replace with pip install
models/
  base.py             # BaseAdapter abstract class — implement this to add a new method
  pta.py              # PTA adapter (original paper method)
  <your_method>.py    # Add new adapters here; expose a module-level build(cfg) factory
PTA_point/
  run_pta.py          # Main entry for point cloud TTA
  utils/utils.py      # Point cloud helpers (mirrors root utils.py pattern)
  scripts/eval_pta.sh # Eval loop for point cloud corruption benchmarks
  dassl/              # Dependency — must be built with `python setup.py develop`
  configs/            # Point cloud YAML configs
  env.yaml            # Full pinned env for point cloud sub-project
```

---

## Dataset Setup

Datasets are **not** included. Expected under `./data/` by default (`--data-root` overrides).

- Image recognition datasets: follow `docs/DATASETS.md` (15 datasets, each needs a specific `split_zhou_*.json` from Google Drive)
- Point cloud datasets: modelnet_c and sonn_c from [HuggingFace](https://huggingface.co/datasets/auniquesun/Point-PRC/tree/main/new-3ddg-benchmarks/xset/corruption)

Dataset name aliases used by `--datasets` / `build_test_data_loader`:
- `I` → ImageNet, `A` → ImageNet-A, `V` → ImageNetV2, `R` → ImageNet-R, `S` → ImageNet-Sketch
- Fine-grained datasets use their full lowercase names (e.g. `caltech101`, `oxford_pets`)

---

## Pre-trained Weights

ULIP weights needed for point cloud eval — **not bundled**:
```
weights/ulip/pointbert_ulip1.pt
weights/ulip/pointbert_ulip2.pt
weights/ulip/slip_base_100ep.pt
```
Download from: https://huggingface.co/datasets/auniquesun/Point-PRC/tree/main/pretrained-weights/ulip

---

## Output

Results are appended (not overwritten) to:
- `outputs/result.txt` — image recognition results
- `outputs/point_result.txt` — point cloud results

The `outputs/` directory must exist before running or the script will crash.

---

## Algorithm Parameters

Hyperparameters live in `configs/<dataset>.yaml`. Only two params matter:
- `alpha` — weight of original text features vs. updated prototype (default: `0.01`)
- `T` — temperature for exponential update weight (default: `20.0`)

Point cloud version hardcodes `T=50` in `PTA_point/run_pta.py`; image version reads `T` from YAML.

---

## Linting / Formatting

No CI, no pre-commit hooks, no test suite. `requirements.txt` pins `flake8==3.7.9`, `yapf==0.29.0`, `isort==4.3.21` — these are available but not enforced by any automation.

---

## Gotchas

- The `clip/` directory is a **local vendored copy** of CLIP. Do not `pip install clip` — the import `import clip` resolves to this local copy.
- `README.md` has a typo: `codna activate pta_point` should be `conda activate pta_point`.
- `build_test_data_loader` in `utils.py` raises a bare string (`raise "Dataset is not..."`) — not a real exception; Python 3 will silently pass this. Add `raise ValueError(...)` if extending dataset support.
- wandb logging is opt-in via `--wandb-log`; omit the flag to run offline.
- `batch_size=1` is hardcoded in the test loaders — TTA processes one sample at a time by design.
