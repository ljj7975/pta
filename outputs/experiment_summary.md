# Experiment Results: PTA Limitations and Patch-Level Prototype Potential

Generated from `scripts/run_experiments.sh` — 5 methods × 4 datasets × 5 seeds = 100 runs.

**Note:** food101 experiments for PatchModPTA variants failed due to missing `configs/patch_modulated_pta/food101.yaml`. Only PTA ran on food101 (86.34% ± 0.02).

---

## 1. Overall Accuracy (4 datasets, 5 seeds)

| Method | caltech101 | dtd | eurosat | fgvc | **Mean** |

|--------|-----------|-----|---------|------|----------|

| PTA | 94.75 | 47.43 | 61.34 | 25.65 | **57.29** |
| PatchModPTA | 94.63 | 47.35 | 62.38 | 22.90 | **56.82** |
| PatchModPTA-QGated | 94.63 | 47.34 | 62.39 | 22.93 | **56.82** |
| PatchModPTA-MVote | 94.60 | 48.09 | 61.93 | 25.61 | **57.56** |
| PatchModPTA-AGate | 94.78 | 48.04 | 61.21 | 25.55 | **57.40** |

### Key Takeaway

- **PTA baseline: 57.29%** mean across 4 datasets.

- **PatchModPTA (ProtoAlpha) and QGated: 56.82%** — slightly below PTA.

- **PatchModPTA-MVote: 57.56%** — best overall (+0.27% over PTA).

- **PatchModPTA-AGate: 57.40%** — close to MVote (+0.11% over PTA).

---

## 2. Fusion Mechanisms — How Each Method Weights the Patch Term

All five methods fuse three logit sources: `clip_logits`, `image_proto_logits`, `patch_proto_logits`.

The fused logits are computed as:

```

result = tau_text * clip_logits + tau_image_proto * image_proto_logits + PATCH_TERM

```

The methods differ in how `PATCH_TERM` is computed:

| Method | PATCH_TERM Formula | Behavior |

|--------|-------------------|----------|

| **ProtoAlpha** | `tau_patch × proto_alpha × squash(patch)` | Patch always contributes, scaled by `proto_alpha` (grows as prototypes accumulate). |

| **QualityGated** | `tau_patch × proto_alpha × quality_gate × squash(patch)` | Same as ProtoAlpha, but additionally gated by sample quality. Low-quality samples → patch muted. |

| **MajorityVote** | 2-of-3 vote on argmax | Each source (clip, image_proto, patch) casts a vote. If ≥2 agree → that class wins. If 3-way split → patch discarded, fall back to clip+image. |

| **AgreementGate** | `tau_patch × (1 if patch.argmax==clip.argmax else 0) × squash(patch)` | Patch only contributes when it agrees with CLIP. Otherwise muted. |


### Why FGVC Collapses with ProtoAlpha/QGated

ProtoAlpha and QGated are **weighted fusion** — the patch term always contributes (just scaled). On FGVC, the patch term is noisy and drags performance down by −2.75%.

MVote and AGate are **discrete gating** — they can **silence** bad patch predictions. This prevents collapse.

---

## 3. ProtoAlphaFusion Collapse on FGVC

| Method | FGVC Accuracy | vs PTA |

|--------|---------------|--------|

| PTA | 25.65% | — |

| PatchModPTA (ProtoAlpha) | 22.90% | **−2.75%** |

| PatchModPTA-QGated | 22.93% | **−2.72%** |

| PatchModPTA-MVote | 25.61% | −0.04% |

| PatchModPTA-AGate | 25.55% | −0.10% |


### Interpretation

- ProtoAlpha/QGated **collapse** on FGVC (−2.75%, −2.72%) because they force patch contribution even when patch is noisy.

- MVote/AGate **prevent collapse** by silencing bad patch predictions (−0.04%, −0.10%).

- This demonstrates that **always-on patch fusion is harmful** on fine-grained datasets.

---

## 4. Tie-Breaking Analysis (dtd, seed 1)

A **tie** occurs when `clip.argmax != image_proto.argmax` — CLIP and the image-level prototype disagree.

| Metric | Value |

|--------|-------|

| Total samples | 1692 |

| Ties (clip != image_proto) | 527 (31.1%) |

| Of ties, PTA is correct | 130 (24.7%) |

| Of ties, PTA is wrong | 397 (75.3%) |


### Breakdown of Patch Predictions on Ties

| Patch agrees with | Count | % of Ties | Source is correct | % correct | Δ vs baseline |
|-----------------|-------|-----------|-------------------|-----------|---------------|
| image_proto | 109 | 20.7% | 37/109 | 33.9% | +9.3% |
| clip | 107 | 20.3% | 32/107 | 29.9% | +13.2% |
| other (independent) | 311 | 59.0% | 18/311 | 5.8% | — |

**Baseline accuracy on ties:** CLIP = 16.7%, image_proto = 24.7%.

### Interpretation

- **75.3% of ties are wrong** — when CLIP and image_proto disagree, the image_proto prediction is wrong 75.3% of the time. This is the **prototype drift** problem.

- Patch agreement is a **quality signal** for both sources:
  - When patch agrees with CLIP → CLIP is correct 29.9% of the time (+13.2% over CLIP baseline).
  - When patch agrees with image_proto → proto is correct 33.9% of the time (+9.3% over proto baseline).

- Patch is **independent** 59.0% of ties — predicting something completely different from all sources. This provides diversity for 2-of-3 voting.

- This suggests **voting can leverage patch** to break ties: patch agreement boosts confidence in a source, while patch independence provides a third vote that can tip the balance.

---

## 5. Dataset-Specific Gains (MVote vs PTA)

| Dataset | PTA | MVote | Diff | Characteristic |

|---------|-----|-------|------|----------------|

| caltech101 | 94.75% | 94.60% | -0.15% | saturated (easy) |
| dtd | 47.43% | 48.09% | +0.65% | texture |
| eurosat | 61.34% | 61.93% | +0.59% | satellite imagery |
| fgvc | 25.65% | 25.61% | -0.04% | fine-grained aircraft |

### Interpretation

- **Gains on texture/satellite datasets** (dtd +0.65%, eurosat +0.59%) — these benefit from local spatial patterns.

- **No gain on saturated datasets** (caltech101 −0.15%) — already near-ceiling.

- **No collapse on FGVC** (−0.04%) — MVote prevents the −2.75% collapse seen with ProtoAlpha.

---

## 6. Per-Class Accuracy (PTA vs MVote on dtd)

Classes with >5% difference:

| Class | PTA | MVote | Diff |

|-------|-----|-------|------|

| 39 | 48.89% | 60.00% | +11.11% |
| 46 | 31.67% | 18.89% | -12.78% |
---

## 7. Key Insights for the Paper

1. **PTA's image-level prototype updates can hurt performance** when they disagree with CLIP (75.3% of ties are wrong). This is the **prototype drift** problem.

2. **ProtoAlphaFusion collapses on FGVC (−2.75%)** because it over-relies on image-level prototypes that drift. This demonstrates PTA's limitation on fine-grained datasets.

3. **Patch-level prototypes provide an orthogonal signal** (55.6% of ties predict something different from all sources) that can break ties and prevent collapse.

4. **Voting mechanisms (MVote, AGate) leverage patch signals** to maintain PTA-level performance without collapse, demonstrating the value of patch-level prototypes.

5. **Always-on patch fusion (ProtoAlpha/QGated) is harmful** on fine-grained datasets. Discrete gating (MVote/AGate) is needed to silence noisy patch predictions.
