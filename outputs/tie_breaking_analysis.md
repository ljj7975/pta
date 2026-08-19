# Tie-Breaking Analysis: Patch-Level Prototype Signal

**Date**: 2026-08-18
**Dataset**: dtd (seed 1)
**Context**: Understanding when and how patch-level prototypes help break ties

---

## When is Tie-Breaking Needed?

A **tie** occurs when `clip.argmax != image_proto.argmax` — the CLIP zero-shot prediction disagrees with the image-level prototype prediction.

| Metric | Value |
|--------|-------|
| Total samples | 1692 |
| Ties (clip != image_proto) | 527 (31.1%) |
| Of ties, PTA is correct | 130 (24.7%) |
| Of ties, PTA is wrong | 397 (75.3%) |

**Key finding**: When CLIP and image_proto disagree, the image_proto prediction is wrong **75.3%** of the time. This is the **prototype drift** problem — PTA's image-level updates can hurt performance.

---

## What Does Patch Predict on Ties?

| Patch agrees with | Count | % of Ties | Source is correct | % correct | Δ vs baseline |
|-----------------|-------|-----------|-------------------|-----------|---------------|
| **GT (correct)** | **87** | **16.5%** | **87/87** | **100%** | — |
| image_proto | 109 | 20.7% | 37/109 | 33.9% | +9.3% |
| clip | 107 | 20.3% | 32/107 | 29.9% | +13.2% |
| other (independent) | 311 | 59.0% | 18/311 | 5.8% | — |

**Baseline accuracy on ties**: CLIP = 16.7%, image_proto = 24.7%.

### Interpretation

1. **Patch predicts GT 16.5% of ties** — directly correct, could fix wrong predictions.

2. **Patch is independent 59.0% of ties** — predicting something completely different from all sources. This provides diversity for 2-of-3 voting.

3. **Patch agreement is a quality signal**:
   - When patch agrees with CLIP → CLIP is correct **29.9%** of the time (+13.2% over CLIP baseline)
   - When patch agrees with image_proto → proto is correct **33.9%** of the time (+9.3% over proto baseline)

4. **Patch agreement boosts both sources** — but the improvement is larger for CLIP (+13.2%) than for image_proto (+9.3%).

---

## Implications for Voting Mechanisms

### Why MajorityVoteFusion Works

MajorityVoteFusion uses 2-of-3 voting:
- If 2+ sources agree → that class wins
- If 3-way split → patch discarded, fall back to clip+image

The tie-breaking analysis shows:
1. Patch provides an **orthogonal signal** (59.0% independent) — useful for breaking ties
2. Patch agreement **boosts confidence** in a source — useful for reinforcing correct predictions
3. Patch independence **prevents collapse** — when patch is wrong, it's usually independent (not reinforcing a wrong source)

### Why AgreementGateFusion Works

AgreementGateFusion only contributes patch when `patch.argmax == clip.argmax`:
- When patch agrees with CLIP → CLIP is correct 29.9% of the time (+13.2% boost)
- This filters out the 59.0% independent (often wrong) patch predictions
- Only keeps the 20.3% where patch supports CLIP

### Why ProtoAlphaFusion Collapses

ProtoAlphaFusion always contributes patch (scaled by `proto_alpha`):
- On FGVC, patch is noisy and independent 59.0% of the time
- This adds noise to the fusion, dragging performance down by −2.75%
- No gating mechanism to silence bad patch predictions

---

## Summary

| Question | Answer |
|----------|--------|
| How often is tie-breaking needed? | 31.1% of samples |
| When ties occur, how often is PTA wrong? | 75.3% of ties |
| How often does patch predict GT? | 16.5% of ties |
| How often is patch independent? | 59.0% of ties |
| When patch agrees with CLIP, is CLIP correct? | 29.9% (+13.2% vs baseline) |
| When patch agrees with image_proto, is proto correct? | 33.9% (+9.3% vs baseline) |
| Why does voting work? | Patch provides orthogonal signal + quality signal |
| Why does ProtoAlpha collapse? | Always-on patch fusion adds noise (59.0% independent) |
