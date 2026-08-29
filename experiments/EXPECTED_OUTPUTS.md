# Expected Outputs from Each Slurm Script

This document describes what each slurm script produces, the analysis it enables, and the exact metrics to expect.

---

## 00_baseline_pta.sh — PTA Baseline

**What it does**: Runs standard PTA (image-level prototype only) on 3 datasets × 4 seeds with per-sample recording enabled.

**Outputs**:
- `outputs/records/PTA-{dataset}-s{seed}/records.jsonl` × 12 files
- Entries appended to `outputs/result.txt`

**What it enables**: Foundation for Part 1 tie-breaking analysis. Provides CLIP logits, image-level prototype logits, and final fused logits for every sample.

**Expected accuracy ranges** (from prior experiments):
| Dataset | Expected Accuracy |
|---------|-------------------|
| dtd | 47.0 – 48.0% |
| oxford_flowers | 74.0 – 75.5% |
| oxford_pets | 90.5 – 91.5% |

---

## 01_part1_tie_breaking.sh — Part 1: Prototype Drift Analysis

**What it does**: Runs `scripts/analyze_ties.py` on PTA baseline records to measure how often CLIP and the image-level prototype disagree, and who is right in those cases.

**Inputs consumed**:
- `outputs/records/PTA-{dataset}-s{seed}/records.jsonl` (from 00)

**Outputs**:
- `outputs/tie_breaking/{dataset}.json` × 3 files

**Metrics produced per dataset**:
```json
{
  "dataset": "dtd",
  "seeds_analyzed": [1, 2, 3, 4],
  "per_seed": {
    "1": {
      "n_total": 1692,
      "n_ties": 527,
      "tie_rate": 31.1,
      "proto_correct_on_ties": 24.7,
      "proto_wrong_on_ties": 75.3,
      "clip_correct_on_ties": 75.3,
      "pta_acc_on_ties": 12.9
    }
  },
  "mean": { ... },
  "std": { ... }
}
```

**What to look for**:
- Tie rate should be ~25-35% across datasets
- Prototype should be wrong ~65-80% of the time on ties (confirms drift hypothesis)
- Results should be consistent across 4 seeds

---

## 02_part2_purity.sh — Part 2.1: Bank Purity

**What it does**: Runs PatchModPTA-CS (default config, CLIP Surgery) on 3 datasets × 4 seeds with per-sample recording. Then runs `scripts/analyze_prototype_purity.py` to compute what fraction of written patches belong to the claimed class.

**Inputs consumed**:
- `outputs/records/PatchModPTA-{dataset}-s{seed}/records.jsonl`

**Outputs**:
- `outputs/records/PatchModPTA-{dataset}-s{seed}/records.jsonl` × 12 files
- `outputs/prototype_purity/{dataset}.json` × 3 files
- Entries appended to `outputs/result.txt`

**Metrics produced per dataset**:
```json
{
  "dataset": "dtd",
  "seeds_analyzed": [1, 2, 3, 4],
  "per_seed": {
    "1": {
      "overall_purity": 62.6,
      "primary_write_purity": 63.4,
      "collateral_write_purity": 46.7,
      "collateral_share": 4.5,
      "total_writes": 12345,
      "primary_writes": 11800,
      "collateral_writes": 545
    }
  },
  "mean": { ... },
  "std": { ... }
}
```

**What to look for**:
- Overall purity should vary: ~62% (dtd) → ~85% (oxford_pets)
- Collateral writes should have much lower purity (38-47%)
- Higher baseline accuracy datasets should have higher purity

---

## 03_part2_separability.sh — Part 2.2: Bank Separability

**What it does**: Runs PatchModPTA-CS with `DUMP_PATCH_BANK` env var to save the final Gaussian bank state. Then runs `scripts/analyze_prototype_separability.py` to measure inter-class vs intra-class cluster similarity.

**Inputs consumed**:
- `outputs/patch_bank_dumps/{dataset}-s{seed}.pt`

**Outputs**:
- `outputs/patch_bank_dumps/{dataset}-s{seed}.pt` × 12 files
- `outputs/prototype_separability/{dataset}.json` × 3 files

**Metrics produced per dataset**:
```json
{
  "dataset": "dtd",
  "seeds_analyzed": [1, 2, 3, 4],
  "per_seed": {
    "1": {
      "intra_class_similarity": 0.738,
      "nearest_cross_class_similarity": 0.859,
      "margin": -0.121,
      "variance_aware": {
        "intra_class_match": 0.593,
        "nearest_cross_class_match": 0.771,
        "margin": -0.178
      }
    }
  },
  "mean": { ... },
  "std": { ... }
}
```

**What to look for**:
- All margins should be **negative** (classes overlap at patch level)
- Oxford_pets should have the most negative margins (fine-grained = more overlap)
- Consistent across seeds

---

## 04_part3_agreement.sh — Part 3.1+3.3: Agreement Signal Validation

**What it does**: Runs `scripts/validate_patch_vote_signal.py` on PatchModPTA records to validate whether CLIP-patch agreement predicts correctness.

**Inputs consumed**:
- `outputs/records/PatchModPTA-{dataset}-s{seed}/records.jsonl`
- `outputs/patch_bank_dumps/{dataset}-s{seed}.pt`

**Outputs**:
- `outputs/patch_vote_validation/{dataset}.json` × 3 files

**Metrics produced per dataset**:
```json
{
  "dataset": "dtd",
  "seeds_used": [1, 2, 3, 4],
  "multiseed": {
    "topk20": {
      "1": {
        "standalone_acc": 41.3,
        "agreement_rate": 60.0,
        "cls_acc_when_agree": 58.4,
        "cls_acc_when_disagree": 23.0,
        "purity_gap": 35.4
      }
    }
  }
}
```

**What to look for**:
- Purity gap (CLIP accuracy when agree vs disagree) should be >30pp for dtd
- Gap should hold across all 4 seeds (seed-invariant signal)
- Topk20 should have highest standalone accuracy on oxford_pets (~65%)

---

## 05_part3_aggregation.sh — Part 3.2: Aggregation Method Comparison

**What it does**: Runs `scripts/analyze_patch_vote_aggregation.py` to compare 13 pooling variants for patch-level signal strength.

**Inputs consumed**:
- `outputs/records/PatchModPTA-{dataset}-s{seed}/records.jsonl` (seed 1 only)
- `outputs/patch_bank_dumps/{dataset}-s1.pt`

**Outputs**:
- `outputs/patch_vote_aggregation/{dataset}.json` × 3 files

**Metrics produced per dataset** (for each of 13 variants):
```json
{
  "variants": {
    "mean": {
      "standalone_acc": 40.8,
      "agreement_rate": 59.6,
      "cls_acc_when_agree": 58.4,
      "cls_acc_when_disagree": 23.0,
      "write_purity_agree": 65.2,
      "write_purity_disagree": 42.1,
      "rescue_rate_when_cls_wrong": 15.3,
      "corrupt_rate_when_cls_right": 8.7
    },
    "topk20": {
      "standalone_acc": 41.3,
      "agreement_rate": 60.0,
      ...
    }
  }
}
```

**What to look for**:
- Topk20 should double standalone accuracy on oxford_pets (30% → 65%)
- Mean should be the baseline for comparison
- Rescue rate should be ~10-15% (patch helps CLIP a little when CLIP is wrong)

---

## 06_part4_trust_signal.sh — Part 4 (NEW): Patch Agreement as Trust Signal

**What it does**: Runs a new `TrustSignal` adapter that implements the trust-signal mechanism:
- When CLIP and patch vote **agree** → report CLIP logits directly
- When CLIP and patch vote **disagree** → report PTA-fused logits (image-level prototype)

Then runs `scripts/analyze_trust_signal.py` to compare this mechanism against baselines.

**Method variants tested**:
1. **TrustSignal-CLIP**: Agree→CLIP, Disagree→PTA
2. **TrustSignal-PTA**: Agree→PTA, Disagree→CLIP (inverted baseline)
3. **TrustSignal-CLIP-Topk20**: Same as #1 but using topk20 aggregation for patch vote

**Outputs**:
- `outputs/records/TrustSignal-{dataset}-s{seed}/records.jsonl` × 12 files
- `outputs/trust_signal_analysis/{dataset}.json` × 3 files
- Entries appended to `outputs/result_trust_signal.txt`

**Metrics produced per dataset**:
```json
{
  "dataset": "dtd",
  "seeds_analyzed": [1, 2, 3, 4],
  "variants": {
    "TrustSignal-CLIP": {
      "per_seed": {
        "1": {
          "overall_acc": 48.2,
          "n_agree": 1015,
          "n_disagree": 677,
          "acc_when_agree": 58.4,
          "acc_when_disagree": 32.9,
          "vs_baseline_pta_delta": 0.7,
          "vs_baseline_clip_delta": 1.2
        }
      }
    }
  },
  "summary": {
    "best_variant": "TrustSignal-CLIP",
    "mean_delta_vs_pta": 0.5,
    "mean_delta_vs_clip": 0.8
  }
}
```

**What to look for**:
- The trust-signal variant should match or slightly exceed PTA on datasets where PTA helps (dtd)
- On high-accuracy datasets (oxford_pets), the variant should avoid PTA's degradation
- The "agree→CLIP, disagree→PTA" direction should outperform the inverted version
- Consistency across seeds is critical

---

## Cross-Script Dependencies

```
00_baseline_pta ──────────────────────────→ 01_part1_tie_breaking
       │
       └── (independent) ──→ 02_part2_purity ──→ 04_part3_agreement
                            │                    └──→ 05_part3_aggregation
                            │
                            └──→ 03_part2_separability
                                 └──→ 04_part3_agreement
                                 └──→ 05_part3_aggregation

06_part4_trust_signal (fully independent)
```

**Parallel execution opportunities**:
- 00 and 02 can run simultaneously (independent data production)
- 01, 03 can run in parallel once their dependencies complete
- 04 and 05 can run in parallel once 02+03 complete
- 06 can run at any time (independent)
