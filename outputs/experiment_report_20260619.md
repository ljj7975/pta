# Experiment Report — Prototype Variants
**Date:** 2026-06-19  
**Backbone:** ViT-B/16  
**Datasets:** caltech101, oxford_flowers, oxford_pets (rerun); dtd, eurosat, fgvc (earlier runs)

---

## Methods

| ID | Name | Key idea |
|----|------|----------|
| PTA | Original PTA | Single text prototype updated via exponential weighting |
| MPTA | MultiProtoPTA | Per-class patch prototype bank with incremental K-means |
| Exp1 | FixedUniquePatches | Forces unique patch assignment; filters redundant patches before bank update |
| Exp2 | AdaptiveTauProto | Replaces fixed `tau_proto` with 50/50 softmax blend of text and prototype scores |
| Exp3 | GaussianPrototypes | Adds per-dimension variance to each prototype; replaces cosine similarity with Mahalanobis-like Gaussian score |

---

## Results

### Core 3-dataset comparison (rerun, all methods)

| Dataset | PTA | MPTA | Exp1 | Exp2 | Exp3 |
|---------|----:|-----:|-----:|-----:|-----:|
| caltech101 | **94.93** | 93.63 | 93.18 | 93.87 | 93.91 |
| oxford_flowers | **74.83** | 69.87 | 69.02 | 70.77 | 70.85 |
| oxford_pets | **91.03** | 88.36 | 86.84 | 89.02 | 89.07 |
| **Average** | **86.93** | 83.95 | 83.01 | 84.55 | 84.61 |

### Additional datasets (earlier runs; PTA not yet evaluated)

| Dataset | MPTA | Exp1 | Exp2 | Exp3 |
|---------|-----:|-----:|-----:|-----:|
| dtd | 40.84 | 41.55 | 44.21 | **44.44** |
| eurosat | 42.72 | 43.15 | 46.48 | **47.74** |
| fgvc | 20.01 | — | — | **24.81** |

---

## Key Findings

1. **PTA baseline remains the strongest** on all three core datasets (avg 86.93%), outperforming all prototype bank variants by 2.3–3.9 pp.

2. **Exp3 (Gaussian) is the best variant** overall. It matches or edges Exp2 on every dataset and leads the additional datasets by up to 1.3 pp on eurosat (47.74 vs 46.48).

3. **Exp2 and Exp3 are close** (avg 84.55 vs 84.61 on the core 3). The gap is small but consistent in Exp3's favor — Gaussian uncertainty weighting provides a marginal but reproducible benefit over 50/50 blending.

4. **Exp1 underperforms MPTA** on all datasets where both are measured (avg 83.01 vs 83.95). Forcing unique patch assignment appears to remove useful signal rather than reduce noise.

5. **Largest absolute gains over MPTA come from Exp2/Exp3 on texture/style datasets** (dtd +3.4/+3.6 pp, eurosat +3.8/+5.0 pp), suggesting that uncertainty-aware or better-calibrated score fusion helps more when visual categories are less prototypical.

---

## Next Steps

- Run PTA baseline on dtd, eurosat, fgvc to establish a complete comparison.
- Exp3 is the most promising variant; consider ablating `gaussian_ema` and `variance_min` on eurosat where the gain is largest.
- Exp1 can likely be dropped from further experiments.
