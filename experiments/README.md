# Systematic PTA Limitation Study — Output Folder Structure

This document describes the expected output folder structure after running all slurm scripts in `experiments/slurm/`.

---

## Root Structure

```
experiments/
├── README.md                              # This file
├── EXPECTED_OUTPUTS.md                    # What each slurm script produces
├── slurm/
│   ├── 00_baseline_pta.sh                 # PTA baseline (Part 1 foundation)
│   ├── 01_part1_tie_breaking.sh           # Part 1: Prototype drift analysis
│   ├── 02_part2_purity.sh                 # Part 2.1: Bank purity analysis
│   ├── 03_part2_separability.sh           # Part 2.2: Bank separability analysis
│   ├── 04_part3_agreement.sh              # Part 3.1+3.3: Agreement signal validation
│   ├── 05_part3_aggregation.sh            # Part 3.2: Aggregation method comparison
│   ├── 06_part4_trust_signal.sh           # Part 4 (NEW): Patch agreement as trust signal
│   └── run_all.sh                         # Convenience: run all scripts sequentially
├── configs/
│   └── trust_signal.yaml                  # Config for the new TrustSignal adapter
└── analysis/
    └── analyze_trust_signal.py            # Part 4 analysis script
```

---

## Output Directories (after running all scripts)

```
outputs/
├── records/                               # Per-sample JSONL records
│   ├── PTA-dtd-s1/records.jsonl           # PTA baseline records
│   ├── PTA-dtd-s2/records.jsonl
│   ├── PTA-dtd-s3/records.jsonl
│   ├── PTA-dtd-s4/records.jsonl
│   ├── PTA-oxford_flowers-s1/records.jsonl
│   ├── ...
│   ├── PTA-oxford_pets-s4/records.jsonl
│   ├── PatchModPTA-dtd-s1/records.jsonl   # PatchModPTA records (Part 2+3)
│   ├── PatchModPTA-dtd-s2/records.jsonl
│   ├── ...
│   ├── PatchModPTA-oxford_pets-s4/records.jsonl
│   ├── TrustSignal-dtd-s1/records.jsonl   # Part 4: Trust signal records
│   ├── ...
│   └── TrustSignal-oxford_pets-s4/records.jsonl
│
├── patch_bank_dumps/                      # Gaussian bank state dumps (.pt files)
│   ├── dtd-s1.pt
│   ├── dtd-s2.pt
│   ├── dtd-s3.pt
│   ├── dtd-s4.pt
│   ├── oxford_flowers-s1.pt
│   ├── ...
│   └── oxford_pets-s4.pt
│
├── result.txt                             # Accumulated accuracy results
├── result_trust_signal.txt                # Part 4 accuracy results (separate file)
│
├── tie_breaking/                          # Part 1 analysis output
│   ├── dtd.json
│   ├── oxford_flowers.json
│   └── oxford_pets.json
│
├── prototype_purity/                      # Part 2.1 analysis output
│   ├── dtd.json
│   ├── oxford_flowers.json
│   └── oxford_pets.json
│
├── prototype_separability/                # Part 2.2 analysis output
│   ├── dtd.json
│   ├── oxford_flowers.json
│   └── oxford_pets.json
│
├── patch_vote_aggregation/                # Part 3.2 analysis output
│   ├── dtd.json
│   ├── oxford_flowers.json
│   └── oxford_pets.json
│
├── patch_vote_validation/                 # Part 3.1+3.3 analysis output
│   ├── dtd.json
│   ├── oxford_flowers.json
│   └── oxford_pets.json
│
└── trust_signal_analysis/                 # Part 4 analysis output (NEW)
    ├── dtd.json
    ├── oxford_flowers.json
    └── oxford_pets.json
```

---

## Naming Conventions

### Record Directories
Pattern: `outputs/records/{METHOD_LABEL}-{DATASET}-s{SEED}/`

- `METHOD_LABEL`: Short identifier for the method variant
  - `PTA`: Standard PTA baseline
  - `PatchModPTA`: Patch-modulated PTA (default fusion)
  - `TrustSignal`: New patch-agreement trust-signal variant
- `DATASET`: Dataset name (e.g., `dtd`, `oxford_flowers`, `oxford_pets`)
- `SEED`: Random seed (1-4)

### Record Files
Each `records.jsonl` contains per-sample records with schema:
```json
{
  "batch_idx": 0,
  "target": 42,
  "pred": 42,
  "correct": true,
  "conf": 0.95,
  "quality_gate": 0.87,
  "gate_mode": "multi",
  "proto_alpha": 1.0,
  "logits": {
    "clip": [0.1, 0.2, ...],
    "image_proto": [0.1, 0.2, ...],
    "patch_proto": [0.1, 0.2, ...],
    "final": [0.1, 0.2, ...]
  },
  "proto_stats": {
    "true": {"n_images": 10, "n_clusters": 5},
    "pred": {"n_images": 8, "n_clusters": 3}
  }
}
```

### Bank Dumps
Pattern: `outputs/patch_bank_dumps/{DATASET}-s{SEED}.pt`

PyTorch saved dict with keys:
- `dataset`: str
- `seed`: int
- `C`: int (number of classes)
- `classes`: list of per-class dicts with keys:
  - `centers`: Tensor [K, D]
  - `variance`: Tensor [K]
  - `appearance`: Tensor [K]
  - `n_images`: int

### Analysis Outputs
Pattern: `outputs/{ANALYSIS_NAME}/{DATASET}.json`

Each JSON file contains structured analysis results. See `EXPECTED_OUTPUTS.md` for the exact schema of each analysis.

---

## Execution Order

The slurm scripts are numbered in dependency order:

1. **00_baseline_pta.sh** — Run first. Produces PTA baseline records needed by Part 1.
2. **01_part1_tie_breaking.sh** — Depends on 00. Analyzes PTA baseline records.
3. **02_part2_purity.sh** — Independent of 00-01. Produces PatchModPTA records.
4. **03_part2_separability.sh** — Can run in parallel with 02 (produces bank dumps).
5. **04_part3_agreement.sh** — Depends on 02's records. Validates agreement signal.
6. **05_part3_aggregation.sh** — Depends on 02's records + bank dumps.
7. **06_part4_trust_signal.sh** — Independent. Runs new TrustSignal adapter.

Scripts 02-03 can run in parallel with 00-01 since they produce independent data.
Scripts 04-05 can run in parallel with each other once 02 completes.
