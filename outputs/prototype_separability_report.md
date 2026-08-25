# Patch-Level Prototype Bank — Inter/Intra-Class Separability Diagnostic

Loads the final per-class Gaussian bank (`centers`/`variance`/`appearance`/`n_images`) dumped via the opt-in `DUMP_PATCH_BANK` env var added to `models/patch_modulated_pta.py` (behavior-neutral, no-op unless set), one seed-1 run per dataset, default `PatchModPTA-CS` (ProtoAlphaFusion) config.

Tests whether a **pure** bank (right-class patches only, per `outputs/prototype_purity_report.md`) is still geometrically **confusable** with neighboring classes — the leading candidate explanation for why patch fusion hurts most on oxford_pets (85.7% write purity, worst accuracy hit) and helps on dtd (62.6% write purity, only dataset with a positive effect).

## All centers

| Dataset | Centers | Classes | Intra-class coherence (mean) | Nearest cross-class confusability (mean/p90) | Background cross-class sim (mean) | Separability margin |
|---|---|---|---|---|---|---|
| dtd | 151 | 39 | 0.667 | 0.846 / 0.950 | 0.621 | -0.178 |
| oxford_flowers | 379 | 93 | 0.688 | 0.921 / 0.986 | 0.680 | -0.233 |
| oxford_pets | 294 | 37 | 0.653 | 0.872 / 0.987 | 0.624 | -0.218 |

## High-appearance-only centers (top half by appearance weight, per class)

| Dataset | Centers | Classes | Intra-class coherence (mean) | Nearest cross-class confusability (mean/p90) | Background cross-class sim (mean) | Separability margin |
|---|---|---|---|---|---|---|
| dtd | 128 | 39 | 0.738 | 0.859 / 0.953 | 0.662 | -0.121 |
| oxford_flowers | 281 | 93 | 0.799 | 0.929 / 0.985 | 0.761 | -0.130 |
| oxford_pets | 194 | 37 | 0.795 | 0.908 / 0.988 | 0.738 | -0.114 |

## Interpretation guide

- **Separability margin** = intra-class coherence − nearest cross-class confusability. Large positive → a class's own centers cluster tighter than its nearest confusor (geometrically separable). Small/negative → even a pure bank cannot distinguish this class from a neighbor at the patch level.
- **Nearest vs. background cross-class similarity**: if nearest ≈ background, there's no *specific* confusable neighbor — similarity is just generic high-dimensional closeness. If nearest ≫ background, specific near-duplicate confusor classes exist (e.g. two visually similar pet breeds).

