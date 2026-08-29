# PatchModPTA: Purity, Separability, and Agreement Analysis

**Scope**: ViT-B/16 backbone, CLIP Surgery, 3 primary datasets (dtd, oxford_flowers, oxford_pets), 4 seeds (1-4). Follow-up to `PTA_Limitations_and_Patch_Signal_Analysis.md` (hereafter "the prior report"). Re-runs the patch-modulation investigation from clean checkpoints.

---

## Background: What This Report Adds

The prior report established that patch-level fusion ("PatchModPTA") never reliably beats plain PTA, and that the *agreement* between CLIP's CLS-level guess and the patch-level vote is a strong but non-actionable quality signal. That report's Part 3 aggregated patch-to-text votes and diagnosed why they don't convert to accuracy, but the write **purity** and **separability** diagnostics it cited came from a small supporting sweep, and it did not test whether a trust signal could be built on those votes.

This report documents the freshly re-run PART 1/2/3 experiments from clean checkpoints:

- **Part 1** — Tie-breaking behavior of the fused patch predictor (where does it help / hurt?).
- **Part 2** — Patch bank write-time **purity** and bank **separability** (recomputed with instrumentation added to the bank itself).
- **Part 3** — Patch-vote **agreement** with CLIP and **aggregation** variants (recomputed, cross-seed).

The central question is whether the *genuinely useful* agreement signal (prior report: +30pp when CLIP and patch vote agree) can be converted into a prediction-time or write-time win — an investigation that continues in Part 4.

### Baseline reference points (4-seed mean)

As orientation for all subsequent comparisons, these are the two reference predictors on the same 3 datasets, pooled over seeds 1-4 (recomputed from the stored PTA records; per-seed values in parentheses):

| Dataset | CLIP zero-shot | PTA fused (`clip + 100×image_proto`) | PTA gain |
|---------|----------------|--------------------------------------|----------|
| dtd | 44.39% | **47.47%** (47.70/47.16/47.81/47.22) | **+3.08pp** |
| oxford_flowers | 71.38% | **74.55%** (74.46/74.34/74.50/74.91) | **+3.17pp** |
| oxford_pets | 89.07% | **91.18%** (91.20/91.22/91.01/91.28) | **+2.11pp** |

PTA consistently gains ~2-3pp over zero-shot CLIP on every dataset. This is the bar that PatchModPTA (Part 1) must clear.

---

## Part 1: Tie-Breaking Behavior of the Patch-Modulated Predictor

### 1.0 Methodology: what a "tie" is, and which predictors are compared

**The two predictors.** Recall from the Background that PTA's final prediction fuses two *base* components: CLIP zero-shot logits plus the image-level prototype logits, i.e. `final_PTA = clip + tau_image × image_proto` (no patch term). `PatchModPTA` adds a third component — the patch-level prototype logits — via `ProtoAlphaFusion`, i.e. `final_PatchModPTA = clip + tau_image × image_proto + patch term`. The two methods therefore share the same two base components but differ in their **final predictor**.

**What a tie is.** A **tie** is a sample on which the two *base* components disagree about the class: `clip.argmax != image_proto.argmax`. Importantly, a tie is defined on the base components (CLIP vs. image-level prototype) and is therefore **identical for PTA and PatchModPTA on the same sample** — the two methods only disagree on *how to break* the tie, because PatchModPTA additionally consults the patch term while PTA does not.

**Which seed.** The tie analysis is run on **one realization (seed 4)**, *not* pooled over seeds. We verified that every figure in the tables below reproduces seed 4's records exactly. Because tie-level statistics fluctuate substantially across seeds (and the PTA↔PatchModPTA delta can even change sign), we additionally report the 4-seed mean so the reader can judge robustness. Throughout, the per-seed sample counts are 1692 (dtd), 2463 (oxford_flowers), and 3669 (oxford_pets).

### 1.1 Tie Rate (seed 4)

| Dataset | Total samples | Ties | Tie rate |
|---------|---------------|------|----------|
| dtd | 1692 | 549 | 32.4% |
| oxford_flowers | 2463 | 421 | 17.1% |
| oxford_pets | 3669 | 320 | 8.7% |

The tie rate is stable across seeds (e.g. dtd 32.2-33.1%, oxford_pets 8.7-9.1%) and consistently higher on the harder dataset (dtd, ~32%) than on the easiest (oxford_pets, ~9%). This means on the hardest dataset roughly **one in three samples is a component-disagreement tie** — i.e., the image-level prototype and CLIP pull in opposite directions — whereas on the fine-grained pets set only ~9% are ties.

### 1.2 Accuracy on Ties: PatchModPTA vs. PTA

For each tie sample we record whether **PTA's final predictor** (clip + image_proto, no patch) and **PatchModPTA's final predictor** (clip + image_proto + patch term) are correct. All accuracies here are on tie samples only.

**Seed 4:**

| Dataset | PTA acc on ties | PatchModPTA acc on ties | Delta |
|---------|-----------------|--------------------------|-------|
| dtd | 25.9% | 28.1% | **+2.2pp** |
| oxford_flowers | 35.6% | 28.3% | **-7.4pp** |
| oxford_pets | 61.2% | 44.1% | **-17.2pp** |

**4-seed mean:**

| Dataset | PTA acc on ties | PatchModPTA acc on ties | Delta |
|---------|-----------------|--------------------------|-------|
| dtd | 26.8% | 28.1% | **+1.3pp** |
| oxford_flowers | 34.0% | 30.5% | **-3.5pp** |
| oxford_pets | 61.0% | 50.3% | **-10.7pp** |

On tie samples the PTA predictor itself is wrong most of the time (dtd 25.9-28.4%, flowers 32.8-35.6%); only oxford_pets keeps a majority (~61%). Adding the patch term helps slightly on dtd but **hurts on both fine-grained datasets**, most sharply on oxford_pets (-10.7pp mean, -17.2pp on seed 4). The per-seed delta range is: dtd −2.0 to +4.4pp, flowers −7.4 to −1.0pp, pets −6.9 to −17.2pp — the *direction* is clear for the fine-grained sets (patch always hurts) but for dtd the sign is not robust across seeds.

### 1.3 Patch Breakdown on Ties

We classify what *effect* the patch term has on each tie sample by comparing **PTA's final prediction** against **PatchModPTA's final prediction** (both relative to the label):

| Outcome | Meaning |
|---------|---------|
| **Helps** | PatchModPTA correct **and** PTA wrong — the patch term *rescued* an otherwise-wrong answer |
| **Hurts** | PatchModPTA wrong **and** PTA correct — the patch term *corrupted* a previously-correct answer |
| **Both right** | Both PTA *and* PatchModPTA are correct (both final predictors arrive at the right class despite the underlying components disagreeing) |
| **Both wrong** | Both PTA and PatchModPTA are wrong |

The **help rate** is the share of *decisive* tie breakdowns in which the patch term did good, `help rate = Helps / (Helps + Hurts)`. It ignores the "Both right"/"Both wrong" cases, where the patch term did not change the outcome. A help rate above 50% means the patch term fixes more ties than it breaks.

**Seed 4:**

| Dataset | Helps | Hurts | Both right | Both wrong | Help rate |
|---------|-------|-------|------------|------------|-----------|
| dtd | 46 | 34 | 108 | 361 | 57.5% |
| oxford_flowers | 26 | 57 | 93 | 245 | 31.3% |
| oxford_pets | 29 | 84 | 112 | 95 | 25.7% |

**4-seed mean help rate:** dtd 54.5% (range 44.0-65.0%), oxford_flowers 39.9% (31.3-46.9%), oxford_pets 31.1% (25.7-36.1%).

**Finding**: When the base components disagree, the patch term only rescues more ties than it breaks on dtd (help rate ~54-58%), and even there the sign is not stable across seeds. On the fine-grained datasets — most dramatically oxford_pets, where the patch term is right fewer than a third of the time on average — the patch term actively corrupts more ties than it fixes. This reproduces the prior report's structural-asymmetry conclusion: patch fusion breaks more than it fixes on high-accuracy baselines; but note the effect on dtd is **not robust across seeds** (help rate 44-65%) and should not be over-interpreted.

---

## Part 2: Patch Bank Quality — Write Purity and Separability

We instrumented the Gaussian patch bank itself (behavior-neutral `DUMP_PATCH_BANK` opt-in) to measure two properties of the *stored* bank, independent of any fusion weight.

### 2.1 Write-Time Purity

#### How patches get written (the gate)

At test time the patch bank is updated from CLIP's own guesses — *no ground-truth label is available*, so a patch is written into class *c*'s bank whenever CLIP's zero-shot softmax probability for class *c* exceeds a threshold. With the default `multi_gate=True` and `conf_threshold=0.3`, an image's patches are written into **every** class whose softmax probability exceeds 0.3 — not just the top-1 predicted class. **Purity** measures, in retrospect (against the held-out ground truth), what fraction of those written patches actually belonged to the claimed class.

#### Metrics (all pooled over 4 seeds)

| Metric | Definition |
|--------|-----------|
| **Writes/image** | Average number of class-bank writes per image (a multi-write image contributes >1) |
| **Multi-write rate** | % of images that wrote to more than one class bank |
| **Zero-write rate** | % of images that wrote to no class bank (all softmax probs were < threshold) |
| **Overall purity** | % of *all* write events whose class matched the true label |
| **Primary-write purity** | % of writes to the *predicted* (top-1 softmax) class that were correct |
| **Collateral-write purity** | % of writes to *additional* (non-top-1) classes that were correct |
| **Collateral share** | fraction of all writes that were collateral (non-top-1) writes |

#### Results (pooled over 4 seeds)

| Dataset | Images | Writes/image | Multi-write rate | Zero-write rate | Overall purity | Primary-write purity | Collateral-write purity | Collateral share |
|---------|--------|--------------|------------------|-----------------|----------------|----------------------|-------------------------|------------------|
| dtd | 6768 | 0.60 | 2.8% | 42.4% | 63.0% | 64.0% | 42.6% | 4.6% |
| oxford_flowers | 9852 | 0.98 | 7.9% | 10.3% | 74.0% | 77.5% | 33.5% | 8.1% |
| oxford_pets | 14676 | 1.08 | 8.9% | 1.2% | 85.6% | 89.9% | 38.4% | 8.4% |

**Per-class median purity**: dtd 66.7%, oxford_flowers 82.4%, oxford_pets 90.9%.

**Finding**: The banks are moderately pure on paper (63-86% overall) but hide a **collateral-write problem**: the additional multi-class writes are far dirtier (33-43%) than primary writes (64-90%). Note also the striking zero-write rates — on dtd 42% of images write *nothing* (their softmax never clears 0.3), so less than two-thirds of the data even reaches the bank. oxford_pets has the *highest overall purity* (85.6%) yet suffers the *worst* fusion hit, reinforcing that purity alone does not determine whether patch fusion helps.

### 2.2 Bank Separability

Even a 100%-pure bank is useless if a class's patches look more like a neighbor's patches than like each other. This diagnostic measures the *geometry* of the stored bank: whether the cluster centers of one class are more similar to each other (intra-class) or to a neighboring class's centers (cross-class).

**Metrics** (cosine similarity between cluster centers):

| Metric | Meaning |
|--------|---------|
| **Intra-class coherence** | mean pairwise similarity between the centers of the *same* class |
| **Nearest cross-class confusability** | similarity between a class's centers and those of its *most similar other class* (mean, plus the p90 across classes) |
| **Background cross-class sim** | mean similarity between a class's centers and all *other* classes' centers (no specific neighbor) |
| **Separability margin** | `intra-class coherence − nearest cross-class confusability`; positive ⇒ a class's own centers cluster tighter than its closest confusor; negative ⇒ not geometrically separable |

We measured this on the finalized bank dump (an analysis-only export; see note below). Two views: all centers, and only the **high-appearance** centers (the most-visited half per class, filtering out rarely-used cluster components).

**All centers:**

| Dataset | Centers | Classes | Intra-class coherence | Nearest cross-class (mean/p90) | Background cross-class sim | **Separability margin** |
|---------|---------|---------|-----------------------|-------------------------------|----------------------------|--------------------------|
| dtd | 157 | 39 | 0.677 | 0.850 / 0.953 | 0.629 | **-0.172** |
| oxford_flowers | 390 | 94 | 0.691 | 0.920 / 0.985 | 0.680 | **-0.228** |
| oxford_pets | 301 | 37 | 0.629 | 0.870 / 0.987 | 0.615 | **-0.240** |

**High-appearance-only centers** (top half by appearance weight per class):

| Dataset | Centers | Classes | Intra-class coherence | Nearest cross-class (mean/p90) | Background cross-class sim | **Separability margin** |
|---------|---------|---------|-----------------------|-------------------------------|----------------------------|--------------------------|
| dtd | 130 | 39 | 0.755 | 0.862 / 0.953 | 0.677 | **-0.107** |
| oxford_flowers | 288 | 94 | 0.800 | 0.930 / 0.985 | 0.763 | **-0.130** |
| oxford_pets | 193 | 37 | 0.749 | 0.907 / 0.989 | 0.719 | **-0.158** |

**Finding**: Every dataset has a **negative separability margin**, on all centers and on high-appearance centers alike. A class's own patch centers are consistently *less similar to each other* than to their nearest neighbor in another class. Even a perfectly-pure bank cannot make patch classes discriminative at the patch level. Because nearest cross-class similarity is well above the generic background similarity (e.g. 0.850 vs 0.629 on dtd), there are *specific* confusable neighbor classes (visually similar fine-grained pairs), not just generic high-dimensional closeness. This is the structural reason patch fusion underperforms.

> **Note**: Separability values are identical across seeds because the analysis operates on the final bank dump state (per-dataset, not per-seed). One `DUMP_PATCH_BANK` run per dataset (seed 1, default `PatchModPTA-CS`) was used.

---

## Part 3: Patch-Vote Agreement and Aggregation (Recomputed)

### 3.0 Methodology: the patch vote and its metrics

**The patch vote.** Each patch of an image can be matched directly to the class text embeddings, *independent of any bank state* — a "patch-to-text zero-shot" vote. This gives a second opinion on the image's class that comes from the **same model and same image** but from local (patch) pooling rather than global (CLS) pooling. This is the signal studied in Part 3.

**Aggregation variants.** A single image has 196 patches; to produce one per-class vector we pool patch-to-class similarities. Background patches dilute mean-pooling, so we tried several variants:

| Variant | Pooling rule |
|---------|--------------|
| `mean` | average over all patches |
| `topk3/5/10/20` | average only over the top-*k* most similar patches |
| `pmean2/4/8/16` | power mean (soft max-style weighting, exponent *p*) |
| `otsu_mean` | average over patches above an automatic (Otsu) similarity threshold |
| `surgery_masked_mean` | average over patches weighted by CLIP Surgery relevance |

**Metrics.** For a given variant:

- **Standalone accuracy** — classification accuracy if we use *only* the patch vote (its argmax over classes), ignoring CLIP's CLS embedding and all prototypes.
- **Agreement rate** — the fraction of samples where the patch vote's argmax equals CLIP's zero-shot argmax (`patch_vote.argmax == clip.argmax`).
- **CLIP acc when agree / disagree** — CLIP's zero-shot accuracy on the subset where the two votes agree vs. disagree. A large gap between them is the "agreement signal."

Because standalone accuracy and agreement rate depend only on each image's own zero-shot CLIP logits and patch embeddings (not on test-set processing order), they are **seed-invariant**; the values below are the 4-seed mean.

### 3.1 The Agreement Signal

The patch embeddings can vote independently of any bank state. We recomputed the agreement signal and its strength. **Every number in 3.1 uses the `mean` pooling variant** from the 3.0 variants table (average over all 196 patches — no top-k, no surgery mask, no p-mean).

**When CLIP and the patch vote (mean-pooled) agree, CLIP is much more likely to be correct** (4 seeds, arithmetic mean):

| Dataset | CLIP acc when agree | CLIP acc when disagree | Gap | Agreement rate |
|---------|---------------------|------------------------|-----|----------------|
| dtd | 58.1% | 23.8% | **+34.3pp** | 60.0% |
| oxford_flowers | 95.1% | 65.2% | **+29.9pp** | 20.6% |
| oxford_pets | 95.4% | 86.4% | **+9.0pp** | 29.9% |

The gap is consistently large on the harder datasets, precisely reproducing the prior report's central meta-signal finding.

### 3.2 Aggregation Variants (Part 3.2)

Mean-pooling all patches is diluted by background. We tested pooling variants (mean, max, top-k mean, soft p-mean) and report standalone accuracy (patch-to-text used alone) and agreement rate.

**dtd (mean rows):**

| Variant | Standalone acc | Agreement rate | cls-acc when agree | cls-acc when disagree |
|---------|----------------|----------------|--------------------|----------------------|
| mean | 40.7% | 60.0% | 58.1% | 23.8% |
| topk20 | 41.4% | 60.6% | 59.0% | 21.9% |
| topk5 | 41.0% | 59.0% | 60.2% | 21.6% |

**oxford_flowers (mean rows):**

| Variant | Standalone acc | Agreement rate | cls-acc when agree | cls-acc when disagree |
|---------|----------------|----------------|--------------------|----------------------|
| mean | 20.9% | 20.6% | 95.1% | 65.2% |
| topk20 | 28.9% | 31.7% | 85.8% | 64.7% |

**oxford_pets (mean rows):**

| Variant | Standalone acc | Agreement rate | cls-acc when agree | cls-acc when disagree |
|---------|----------------|----------------|--------------------|----------------------|
| mean | 30.1% | 29.9% | 95.4% | 86.4% |
| topk20 | **65.4%** | **64.8%** | 95.6% | 77.1% |

**Key finding (reproduced)**: `topk20` — averaging the top-20 most patch-similar patches — roughly doubles standalone accuracy on oxford_pets (30% → 65%) by focusing on the most relevant patches, but still falls below the CLIP baseline there and far below it on oxford_flowers. The pooling improvement is real but bounded by the underlying patch features.

**Detail (dtd, mean variant):** `standalone_acc=40.66`, `agreement_rate=60.05`, `cls_acc_when_agree=58.07`, `cls_acc_when_disagree=23.82`; write-purity when agree `71.98%` vs. when disagree `43.71%`; rescue rate (patch vote correct on samples where CLIP is wrong) `10.41%`, corrupt rate (patch vote wrong on samples where CLIP is right) `21.44%`.

### 3.3 Cross-Signal: Confidence × Agreement Quadrants

We crossed prediction confidence with agreement to check whether the agreement signal adds information *within* both confidence regimes, and how write purity varies.

**Definitions.** **Confidence** is the CLIP write margin, `softmax_top1 − softmax_top2` (the gap between the top two class probabilities): a sample is **confident** if this margin `≥ 0.2`, else **ambiguous**. **Agreement** is whether the patch vote's argmax equals CLIP's argmax. Crossing the two gives four quadrants. **Write purity** here is the Part 2.1 metric evaluated per quadrant (fraction of writes into the true class).

**Write purity by quadrant (mean variant):**

| Dataset | Confident & agree | Confident & disagree | Ambiguous & agree | Ambiguous & disagree |
|---------|-------------------|----------------------|-------------------|----------------------|
| dtd | 75.2% | 48.8% | 47.3% | 35.3% |
| oxford_flowers | 97.4% | 75.8% | 73.9% | 41.2% |
| oxford_pets | 97.2% | 89.2% | 58.8% | 39.6% |

Write purity is highest when the two signals agree **and** CLIP is confident, and degrades sharply toward ambiguous-disagree — so agreement carries information beyond confidence alone. The absolute level is bounded by dataset difficulty (dtd is uniformly dirtier; oxford_pets is clean except when ambiguous).

**Confusability margin by agreement (mean variant):** the separability margin stays **negative regardless of whether the two signals agree** (e.g. dtd: -0.277 agree vs -0.247 disagree; oxford_pets: -0.357 vs -0.337; oxford_flowers: -0.295 vs -0.310). Agreement does *not* imply that the underlying patch classes become separable — the negative-margin problem in Part 2.2 is intrinsic, not a byproduct of disagreement.

### 3.4 Qualitative Examples (topk20)

Concrete cases illustrate what the patch vote does. Notation below is `target → CLIP-prediction → patch-prediction`; a **rescue** is a sample where CLIP was wrong (`CLIP-prediction ≠ target`) but the patch vote was right (`patch-prediction = target`); a **corrupt** is one where CLIP was right but the patch vote was wrong.

**Rescue** (patch correct, CLIP wrong) — dtd: `smeared`→`crystalline`→`smeared`, `striped`→`banded`→`striped`; oxford_flowers: `rose`→`camellia`→`rose`, `artichoke`→`spear thistle`→`artichoke`; oxford_pets: `bengal`→`abyssinian`→`bengal`, `persian`→`ragdoll`→`persian`.

**Corrupt** (CLIP correct, patch wrong) — dtd: `grid`→`grid`→`waffled`, `stained`→`stained`→`smeared`; oxford_flowers: `primula`→`primula`→`wild pansy`, `anthurium`→`anthurium`→`wild pansy`; oxford_pets: `german_shorthaired`→`german_shorthaired`→`beagle`, `wheaten_terrier`→`wheaten_terrier`→`pug`.

**Counts (mean variant, seed 1):** rescue vs. corrupt — dtd 98/161; oxford_flowers 31/**1275**; oxford_pets 59/**2221**. The patch vote rescues a few real cases but corrupts many more; the imbalance is most extreme on the fine-grained datasets — precisely where the accuracy cost in Part 1.2 is largest.

---

## Part 4a: Write-Rule Study (Gating the Image-Prototype Write)

### 4a.0 Introduction / motivation

Parts 1–3 established that the *agreement signal* (CLIP CLS-guess vs. patch-vote) is a strong, reproducible correctness cue, but that it does **not** convert to an accuracy win when fused into the *prediction*. Part 4 asks the complementary question: can the signal help at the *write* step instead of the prediction step?

This Part 4 is a **PTA-only** study: the `patch_proto` bank is deliberately not used at all (nothing to do with PatchModPTA). We hold the **prediction rule fixed** across every variant — `final = clip + 100 × image_proto`, argmax (identical to base PTA) — and vary only the **write rule**: *which images we allow to write into the image prototype, and for which class*. Any accuracy difference between variants is then solely attributable to accumulated prototype-bank differences. This directly tests whether a write-time trust gate on the image prototype can beat always-writing PTA.

### 4a.1 The four write-gate modes

The write is a single-class EMA of the top-1 `argmax(clip)` image embedding into that class's prototype row (the Part 4 write rule — a TPT-style hard top-1 write; see 4a.3 for how this differs from the base PTA multi-class write). Only whether / how often the write fires differs:

| write_gate | write condition | effect |
|------------|-----------------|--------|
| `baseline` | always write `argmax(clip)` | the new top-1 write control |
| `confident` | write iff `softmax(clip).top1 − .top2 ≥ 0.2` | drop low-confidence writes |
| `agree` | write iff `patch_vote(topk20) == argmax(clip)` | drop writes where the patch vote disagrees |
| `confident_and_agree` | write iff both `confident` AND `agree` | drop writes on the joint condition |

Signals are causal (frozen CLIP logits + stateless topk20 patch vote, no bank read). The agree signal comes from `utils.patch_vote.compute_patch_vote(aggregation="topk20")` — the Part 3.3 topk20 finding — and deliberately does **not** touch the `patch_proto` bank.

### 4a.2 Results (4-seed mean; per-seed in parentheses)

| write_gate | dtd | oxford_flowers | oxford_pets |
|------------|-----|----------------|-------------|
| CLIP zero-shot (ref) | 44.39 | 71.38 | 89.07 |
| PTA fused (ref) | 47.47 | 74.55 | 91.18 |
| `baseline` (top-1 write) | **46.94** (46.93/46.57/47.34/46.93) | **74.54** (74.95/74.46/74.22/74.54) | **90.86** (91.01/90.95/90.76/90.73) |
| `confident` | 42.48 (42.14/42.73/42.43/42.61) | 74.12 (73.61/74.34/73.73/74.79) | 90.36 (90.38/90.30/90.27/90.49) |
| `agree` | 43.39 (43.38/42.20/44.86/43.14) | **42.75** (43.20/43.16/42.18/42.47) | 86.06 (86.26/86.05/85.99/85.96) |
| `confident_and_agree` | 41.38 (41.73/40.66/41.90/41.25) | **42.07** (42.67/42.10/41.70/41.82) | 85.48 (85.83/85.25/85.53/85.31) |

Reference rows are the stored PTA records (Part 1 baseline numbers); the four write-gate rows are the new Part 4a runs (seed 1-4, `outputs/result_write_gate.txt`).

**Observations:**

1. **The new top-1 baseline is a fair PTA match.** `baseline` (single top-1 write, always) lands within ~0.0–0.5pp of the base PTA fused reference on every dataset (dtd −0.5pp, flowers −0.0pp, pets −0.3pp). So redefining the write as a single top-1 write — rather than base PTA's multi-class `w ≥ 0.1` write — costs almost nothing, and it gives us the cleanest possible control class for the gate comparison.
2. **Gating by confidence alone barely changes things** (`confident` ≈ baseline, within noise, since the `w ≥ 0.2` confident subset already covers most correct writes on the clean fine-grained sets: write rate 0.77 flowers, 0.91 pets).
3. **Gating by agreement is sharply harmful on fine-grained sets.** `agree` and `confident_and_agree` collapse `oxford_flowers` to ~42-43% (from the 74.5% baseline) and knock ~5pp off `oxford_pets` (90.9→85.5-86.1). On dtd the effect is much smaller (≈3-5pp).

### 4a.3 Why agreement-gating hurts (the mechanism)

The gate is *correct* — records confirm writes fire exactly when the specified condition holds — but the underlying agree signal is too weak to be a useful *write* gate. Write rates (seed 1):

| write_gate | dtd | oxford_flowers | oxford_pets |
|------------|-----|----------------|-------------|
| `baseline` | 1.00 | 1.00 | 1.00 |
| `confident` | 0.50 | 0.77 | 0.91 |
| `agree` | 0.61 | **0.32** | 0.65 |
| `confident_and_agree` | 0.42 | **0.28** | 0.62 |

And the standalone signal accuracy (seed 1) versus CLIP:

| dataset | topk20 pv acc | clip acc |
|---------|---------------|----------|
| dtd | 0.414 | 0.449 |
| oxford_flowers | 0.289 | 0.432 |
| oxford_pets | 0.654 | 0.863 |

The topk20 patch vote is **never more accurate than CLIP** on these datasets, and on `oxford_flowers` it is far weaker (0.29 vs 0.43). The `agree` gate therefore discards a large share of the data — on flowers the patch vote disagrees with clip on ~68% of images, dropping the write rate to ~0.3. Because the patch vote disagrees with CLIP precisely when CLIP is often *right*, the gate starves the prototype of exactly the confident writes it needs to adapt; the frozen classification then leans back toward zero-shot. This is the write-side mirror of Parts 1–3: the agreement signal does not identify *mistakes* reliably enough to make a strict write gate pay off. (This is exactly why Part 4b tries the *soft* version — never drop a write, only up-weight the trusted ones.)

> Minor note on scope: base PTA writes every class with `softmax ≥ 0.1` (a multi-class write), whereas all Part 4a gates write only the single top-1 class. The always-write top-1 `baseline` already tracks the base-PTA reference to within 0.5pp, so this write-rule narrowing is not the source of the gate differences — the gate *drop-rate* is.

---

## Part 4b: Write-Weight Study (Re-weighting Writes by Trust: Up-Boost, Down-Weight, Two-Sided)

### 4b.0 Motivation

Part 4a showed that *dropping* writes by agreement (the hard gate) hurts — the topk20 agree signal is weakly discriminative but the raw write *weight* in PTA is small, so gating is all-or-nothing. Part 4b tests the **soft** counterpoint: **never drop a write; instead up-weight the writes we are most confident about.**

Concretely, the **write rule is identical to base PTA** — every image writes `argmax(clip)` into the image prototype (single-class EMA, same top-1 write as 4a `baseline`, no write ever dropped). The only change is the **write weight**: when an image is both **confident** (`softmax(clip).top1 − .top2 ≥ 0.2`) **and** the topk20 patch vote **agrees** with `argmax(clip)`, the EMA weight is multiplied by a boost factor `B` (clamped to ≤ 1.0), pulling the prototype harder toward images we trust.

This isolates the question the hard gate could not: *is confidence-weighted (as opposed to all-or-nothing) evidence actually useful at write time?* The prediction rule is unchanged (`clip + 100 × image_proto`), so any gain is attributable purely to how the weight redistributes adaptation effort toward trusted images.

### 4b.1 Design

- **Method** `reweight_pta` (`models/reweight_pta.py`), config `configs/reweight_pta/` (inherits `base`).
- **Write rule:** always `argmax(clip)`, single-class EMA with weight `w_new = 1 − exp(−w_top1 / T)` (exactly base PTA's formula for the written class; `T = 20`). When `confident AND agree`: `w_new ← (w_new × B).clamp(max=1.0)`; otherwise `B = 1` (standard weight).
- **Gate/agree signal:** same as 4a — `conf_margin_thresh = 0.2`, topk20 `compute_patch_vote`.
- **Boost sweep:** `B ∈ {1, 2, 5, 10}`. `B = 1` reproduces the 4a `baseline` exactly and serves as the in-grid control (a no-boost re-run, so the sweep isolates the boost effect cleanly).
- **Grid:** 4 boosts × 3 datasets × 4 seeds = **48 tasks** (slurm `experiments/slurm/07_part4_reweight.sh`), mirroring Part 4a. Outputs to `outputs/result_reweight.txt` and per-run records under `outputs/records_rw/` (each record logs the applied `boost`, `clip_margin`, `patch_vote_pred`).
- **Prediction:** `clip + 100 × image_proto` (WeightedFusion), identical across all boost levels.

### 4b.2 Metric

The primary comparison is **accuracy vs boost level** on each dataset (4-seed mean), against the `B = 1` baseline and the 4a numbers. Since boosting raises the effective EMA weight on the trusted subset:

- If the agree signal is *useful*, larger `B` should lift accuracy above the `B = 1` baseline on some dataset — most plausibly the fine-grained sets, where the trusted (confident-and-agree) subset is highly pure (Part 3.3 quadrant purity 97%).
- If, as Part 4a suggests, the agree signal only *coincides* with (rather than causes) correctness, boosting will be accuracy-neutral or slightly harmful (over-weighting a prototype direction derived from a noisy subset).

Supporting data: `outputs/result_reweight.txt` (48 lines = 4 boosts × 3 datasets × 4 seeds), `outputs/records_rw/RW-b{1,2,5,10}-{dataset}-s{1..4}/`.

### 4b.3 Results (4-seed mean)

| Boost | dtd | flowers | pets | avg (3) |
|:-----:|----:|--------:|-----:|--------:|
| **B=1** (== base PTA control) | **46.943** | **74.543** | **90.862** | **70.783** |
| B=2   | 46.735 | 74.595 | 90.838 | 70.723 |
| B=5   | 46.542 | 74.412 | 90.663 | 70.539 |
| B=10  | 46.262 | 74.230 | **89.873** | 70.122 |

Per-seed spread is ~0.4–0.8pp on each dataset (e.g. pets seed range at B=1: 90.73–91.01; at B=10: 89.64–90.02).

Per-seed values (seed 1–4) for reference:

| Boost | dtd | flowers | pets |
|:-----:|-----|---------|------|
| B=1 | 47.34, 46.93, 46.93, 46.57 | 74.22, 74.46, 74.54, 74.95 | 90.76, 90.73, 90.95, 91.01 |
| B=2 | 46.75, 46.75, 46.93, 46.51 | 74.30, 74.30, 74.83, 74.95 | 90.71, 90.60, 91.01, 91.03 |
| B=5 | 46.51, 46.75, 46.63, 46.28 | 74.18, 74.14, 74.62, 74.71 | 90.54, 90.57, 90.73, 90.81 |
| B=10| 46.75, 46.28, 46.22, 45.80 | 74.14, 74.06, 74.38, 74.34 | 90.02, 89.64, 90.02, 89.81 |

### 4b.4 The boost hurts: mechanism

**Boosting does not help — it degrades accuracy monotonically with `B`.** Every boost level sits below the `B = 1` baseline on the 3-dataset average (B=2 −0.06, B=5 −0.24, B=10 −0.66 pp), and the degradation is *ordered*: the bigger the boost, the larger the drop. `dtd` (46.94→46.26) and `pets` (90.86→89.87) both fall strictly with `B`; `flowers` shows only a noise-level bump at B=2 (+0.05pp, well inside its ~0.5pp seed spread) before falling. The strongest move is fine-grained `pets` at B=10, down **0.99 pp** from baseline.

Mechanistically this is the mirror image of Part 4a, arriving at the same conclusion from the soft side. The boost re-weights the EMA toward the confident-and-agree subset, but it does not change *which* class the prototype drifts toward — it only makes the prototype lean harder on a subset that is pure (Part 3.3, 97%) yet has **no causal relationship with correctness**. Over-weighting that subset makes the single-class EMA more sensitive to the recent, idiosyncratic trusted samples (a hot-start momentum effect), so the prototype path is *less* stable, not more. The default boost of 1 — i.e. the standard EMA weight — is already at the optimum; the trust signal is informative as a *diagnostic* (Part 3.1) but provides no usable lever at write time, whether applied as a hard gate (4a) or as a weight multiplier (4b).

**Conclusion (negative):** there is no boost level that improves on the `B = 1` baseline on the 3-dataset dev set, and no dataset where boosting helps beyond seed noise. Scaling the write weight by the trust signal is not a viable adaptation mechanism.

### 4b.5 Down-weighting the untrusted writes (two-sided re-weighting, Part 4b extension)

Because up-boosting alone (4b.3–4b.4) only *amplified* the trusted subset and still hurt, the natural extension — motivated by the user's hypothesis that *reducing* the impact of low-confidence/disagreeing writes should converge faster than base — is **two-sided re-weighting**: up-weight the **trusted** writes (confident AND agree) while simultaneously **down-weighting the untrusted** ones (ambiguous OR disagree). Method `reweight_pta` (extended in `models/reweight_pta.py`); the write rule is again byte-identical to base PTA (always `argmax(clip)`, single-class EMA, never dropped); only the EMA weight scales with trust:

$$\text{applied weight} = \begin{cases} \text{boost} & \text{if confident AND agree (trusted)} \\ \text{boost\_down} & \text{if ambiguous OR disagree (untrusted)} \\ \end{cases}$$

`boost = 1, boost_down = 1` is exactly base PTA (in-grid control). Sweep = 5 (boost, boost_down) settings × 3 datasets × 4 seeds = **60 tasks** (`experiments/slurm/08_part4b_downweight.sh`, records under `outputs/records_dw/`, results in `outputs/result_reweight_down.txt`): settings `(1.0,1.0)`, `(1.0,0.5)`, `(1.0,0.1)`, `(2.0,0.5)`, `(2.0,0.1)`. Prediction rule kept identical (`clip + 100 × image_proto`). This run additionally records a per-sample **convergence trajectory** (online cumulative accuracy vs. samples seen) in each task's `convergence.json` (`utils/records.py: write_convergence`) so the convergence-speed hypothesis can be tested directly, not just final accuracy.

#### Accuracy (4-seed mean)

| (boost, boost_down) | dtd | flowers | pets | avg (3) |
|:-----:|----:|--------:|-----:|--------:|
| **(1.0, 1.0)** == base PTA control | **46.943** | **74.543** | 90.862 | **70.783** |
| (1.0, 0.5) | 46.795 | 74.453 | **90.950** | 70.733 |
| (1.0, 0.1) | 45.922 | 72.735 | 90.343 | 69.667 |
| (2.0, 0.5) | 46.470 | 74.227 | 90.933 | 70.543 |
| (2.0, 0.1) | 45.538 | 72.515 | 90.045 | 69.366 |

Mild down-weighting `(1.0,0.5)` is the only setting that is essentially accuracy-neutral versus the control (avg −0.05pp, −0.15 on dtd, −0.09 on flowers, +0.09 on pets — all inside the ~0.5pp seed spread). Aggressive down-weighting `down=0.1` costs **−1.1 to −1.4 pp** on the average (with and without up-boost), concentrated on `flowers` (74.54→72.74, −1.81pp) and `dtd` (46.94→45.92, −1.02pp); `pets` degrades only mildly (90.86→90.34). Adding up-boost on top `(2.0,·)` never recovers the loss — it compounds it, matching 4b.3.

#### Convergence (robust measures)

The convergence-speed hypothesis was tested with the two **seed-noise-robust** metrics in `convergence.json` — early-fraction online accuracy and normalized AUC (area under the online-acc trajectory), averaged over all 12 runs per setting (3 datasets × 4 seeds):

| (boost, boost_down) | acc @ 10% | acc @ 25% | acc @ 50% | auc_norm | final |
|:-----:|-----:|-----:|-----:|-----:|-----:|
| (1.0, 1.0) | **68.83** | **69.66** | **69.84** | **0.9805** | 70.78 |
| (1.0, 0.5) | 67.69 | 69.32 | 69.69 | 0.9772 | 70.73 |
| (1.0, 0.1) | 61.12 | 66.08 | 67.84 | 0.9483 | 69.67 |
| (2.0, 0.5) | 67.50 | 69.18 | 69.46 | 0.9767 | 70.54 |
| (2.0, 0.1) | 60.78 | 65.80 | 67.47 | 0.9482 | 69.37 |

**Down-weighting does not speed up convergence — it slows it.** Both robust convergence signals degrade *monotonically* as `boost_down` shrinks: early (10%-seen) online accuracy falls from **68.83 → 67.69 → 61.12** and normalized AUC from **0.9805 → 0.9772 → 0.9483**. The apparent `t_to_80pct` speedup at `down=0.5` (3.6 vs. 13.2 samples) is an artifact of that metric: `t_to_80pct` is dominated by the first few (stochastic) samples on the single hardest dataset (`dtd` seed-1 alone has `t80 ∈ {5, 121, 125}` across the three `down` levels while the other three seeds sit at `t80 ∈ {1, 2}` at every setting), so it swings wildly between settings while the robust AUC/early-fraction curves move monotonically *against* the hypothesis. In other words, the untrusted-down-weight corollary of 4b.4 fails the same way the trusted-up-weight did: it does not make the prototype reach its peak earlier, and at any meaningful strength it simply lowers the ceiling.

### 4b.6 Two-sided re-weighting: mechanism & conclusion

Mechanistically, the two-sided operator is the symmetric failure of the trust signal at write time. The untrusted subset (ambiguous OR disagree) is not *useless* — it is just heterogeneous: some of it is genuinely incorrect (so down-weighting helps a little, the `(1.0,0.5)` effect on `pets`), but much of it is *correct-by-coincidence* or *correct-but-noise-rounding* (Part 3.3 betrays the signal: even the disagree quadrant is not cleanly wrong). Suppressing it therefore discards proportionally more correct writes than incorrect ones, exactly like the 4a hard gate but attenuated. Because the written class is fixed regardless (always `argmax(clip)`), shrinking the weight only shrinks how much each untrusted sample nudges the prototype — and since those nudges are roughly balanced-correct/incorrect, the net is a slight *slow* of the prototype's drift (lower early accuracy) with no compensating gain in final peak. There is no `(boost, boost_down)` in the grid that beats the `(1.0, 1.0)` base on final accuracy *or* on any robust convergence measure.

**Conclusion (negative, extends 4b.4):** the two-sided write-weight operator is not a viable adaptation lever either. Up-weighting trusted writes hurt (4b.3), down-weighting untrusted writes does not help convergence and at `down ≤ 0.5` costs accuracy, and combining the two never recovers the loss. The trust axis (confidence × agreement) discriminates *prediction correctness* well (Part 3.1) but provides no usable *write-time* lever along any scalar direction — up, down, or both. The user's convergence-speed hypothesis is disconfirmed on robust metrics: the prototype does not reach its peak accuracy earlier under re-weighting, it just peaks lower (or no higher).

---

## Summary

| Finding | Implication |
|---------|-------------|
| PatchModPTA rescues ties on dtd (help rate ~54-58%) but corrupts fine-grained ties (flowers ~40%, pets ~31%); dtd effect is not robust across seeds (44-65%) | Patch fusion consistently breaks more high-accuracy (fine-grained) decisions than it fixes |
| Write purity is moderate (63-86%) but collateral writes are far dirtier (33-43%) | Dirt concentrates in the multi-class writes |
| Separability margins are negative on every dataset and every variant (-0.11 to -0.24) | Even a pure bank cannot discriminate classes at patch level |
| Agreement signal confirms as strong +33-35pp on hard sets, small +9pp on easy sets | Confirms the prior report's meta-signal, reproduced |
| Write-gating the image prototype by agreement (Part 4a) collapses fine-grained accuracy (flowers 74.5→42.1-42.8, pets 90.9→85.5-86.1) because the topk20 patch vote is never more accurate than CLIP (e.g. flowers 0.29 vs 0.43) and gates away correct writes | The agreement signal is not discriminative enough to *drop* writes |
| Boosting the write weight on confident-and-agree images (Part 4b) also fails: accuracy degrades monotonically with boost level (B=1 70.78 → B=10 70.12 avg; pets drops 0.99pp at B=10) | The trust signal is informative as a *diagnostic* but provides no usable lever at write time — neither as a hard gate (4a) nor as a weight multiplier (4b); the standard EMA weight (boost=1) is already optimal |
| Two-sided re-weighting (boost trusted + down-weight untrusted, Part 4b extension) fails on final accuracy **and** on convergence: `down∈{0.5,0.1}` monotonically lowers AUC (0.9805→0.9772→0.9483) and early online-acc (10%: 68.83→67.69→61.12); `(1.0,0.5)` is the only accuracy-neutral setting and it does not converge faster (t80 "speedup" is a seed-1 early-sample artifact) | The user's convergence-speed hypothesis is disconfirmed: re-weighting (up, down, or both) does not make the prototype reach peak accuracy earlier — it peaks lower or no higher. No scalar on the trust axis (up/down/both) improves on base PTA, so adaptation effort stays at the default EMA weight |

The re-run confirms and sharpens the prior report: patch-level prototypes are structurally non-separable (Part 2.2), patch-level fusion does not reliably convert the agreement signal into wins, and a hard *write-time* gate on the image prototype built from that signal fails for the same reason (Part 4a) — on every dataset the topk20 patch vote is at best equal to, and often far weaker than, CLIP itself, so any rule that suppresses writes whenever the two signals disagree starves the prototype of exactly the confident writes it needs. Part 4b tested the soft counterpoint (never drop a write, only up-weight the confident-and-agree ones) and arrives at the same conclusion: boosting the write weight by B ∈ {2,5,10} degrades accuracy monotonically versus the B=1 control (pets 90.86→89.87 at B=10), because over-weighting the trusted subset makes the single-class prototype EMA *less* stable without changing which class it drifts toward. The two-sided extension (4b.5–4b.6) then tested the remaining direction — down-weighting the *untrusted* (ambiguous-or-disagree) writes, hypothesized to converge faster than base — and it fails too: on robust convergence measures (normalized AUC 0.9805→0.9772→0.9483 and 10%-seen online accuracy 68.83→67.69→61.12 as `boost_down` goes 1.0→0.5→0.1) the prototype converges *slower*, not faster, and at any non-trivial strength it also lowers final accuracy (avg 70.78→70.73→69.67; flowers 74.54→72.74 at `down=0.1`). The untrusted subset is heterogeneous — some genuinely wrong, much of it correct-by-noise-rounding — so suppressing it discards roughly as many correct writes as incorrect ones. The confidence×agreement signal is well-validated as a *meta-signal* (Parts 1–4) but gives no usable write-time lever along *any* scalar direction — as a gate it starves the prototype, as an up-weight it destabilizes it, as a down-weight it slows it. Adaptation effort is best left at the default PTA write.

---

## Appendix: Supplementary Data

### A.1 Mean Accuracies (4 seeds) — Background, Part 1/3

| Method | dtd | oxford_flowers | oxford_pets | Mean |
|--------|-----|----------------|-------------|------|
| CLIP zero-shot (baseline) | 44.39 | 71.38 | 89.07 | 68.28 |
| PTA (baseline) | 47.47 | 74.55 | 91.18 | 71.07 |
| PatchModPTA (default / -CS) | 47.46 | 72.21 | 89.10 | 69.59 |

### A.2 Tie-Breaking Counts (seed 4)

These are the raw counts behind Section 1.3 for the reported seed (seed 4).

| Dataset | Ties | Helps | Hurts | Both right | Both wrong |
|---------|------|-------|-------|------------|------------|
| dtd | 549 | 46 | 34 | 108 | 361 |
| oxford_flowers | 421 | 26 | 57 | 93 | 245 |
| oxford_pets | 320 | 29 | 84 | 112 | 95 |

---

*This document is self-contained. Data of the `outputs/` results — `result.txt` (Part 1), `tie_breaking/`, `prototype_purity/`, `prototype_separability/`, `patch_vote_validation/`, `patch_vote_aggregation/` (Parts 2-3), `result_write_gate.txt` + `records_wg/` (Part 4a), `result_reweight.txt` + `records_rw/` (Part 4b up-weight), `result_reweight_down.txt` + `records_dw/` incl. per-run `convergence.json` (Part 4b two-sided/down-weight). See `METHODOLOGY.md` for how each metric is derived.*
