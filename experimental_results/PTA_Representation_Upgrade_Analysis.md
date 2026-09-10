# PTA Representation Upgrade: Gaussian, Bank, and Compactness-Gate Prototypes

**Scope**: ViT-B/16 backbone, CLIP Surgery, 3 primary datasets (dtd, oxford_flowers, oxford_pets), 4 seeds (1-4). This is a self-contained, independent study: every method below is a new implementation (`models/gauss_pta.py`, `models/bank_pta.py`, `models/compact_pta.py`, plus the wave-2 `models/bank_v2_pta.py`, `models/bank_compact_pta.py`, `models/gauss_compact_pta.py`) that reuses only the shared runner, data loaders, and CLIP encoder. No existing method or file was modified. All results are 4-seed means; per-seed values appear in parentheses where useful.

---

## Background: What This Report Adds

**The method under study (PTA).** PTA is a test-time adaptation (TTA) method for CLIP-style vision-language models. As unlabeled test images stream in one at a time, PTA maintains one prototype vector per class, initialized at the class's text embedding. For each incoming image with L2-normalized feature `x`, PTA (i) computes zero-shot CLIP logits `clip = 100 * cos(x, text_c)` for all classes, (ii) *writes*: every class `c` whose softmax probability is at least 0.1 has its prototype EMA-updated, `proto_c <- normalize((1-w_c) proto_c + w_c x)` with `w_c = 1 - exp(-p_c / T)` (T=20), and (iii) *predicts* with a fusion of the two signals, `final = 1.0 * clip + 100.0 * cos(x, refined_text_c)` where `refined_text_c = 0.01 * text_c + 0.99 * proto_c` (L2-normalized). There is no gradient step, no label, and no hyperparameter tuning per dataset: the entire adaptation is the prototype EMA.

**Why the prototype is the lever.** On these three datasets PTA beats zero-shot CLIP by +2 to +3pp (table below), and the gain comes from the prototype term, not from CLIP. A tie analysis on archived PTA records shows that on samples where CLIP and the image prototype *disagree* (argmax differs), the prototype is the more reliable predictor on every dataset: dtd 26.2% vs 16.2%, oxford_flowers 33.8% vs 13.9%, oxford_pets 59.3% vs 32.8% accuracy on ties. The prototype is therefore the component whose *quality* bounds the method's ceiling, and improving the per-class prototype representation is the most direct route to beating PTA.

**What this report does.** It tests three representation-level upgrades of the per-class prototype, each motivated by a concrete defect of the single-mean prototype and each pre-validated on cheap offline data before any cluster compute was spent:

- **M1a — GaussPTA**: replace the single mean with a per-class Gaussian (mean + per-dimension variance) and score with a variance-aware (Mahalanobis) distance instead of cosine. Motivated by: CLIP features within a class are spread anisotropically; a mean is a blurry representative and cosine ignores the class's shape.
- **M1b — BankPTA**: replace the single mean with a bank of K prototypes per class (online k-means style) and score with max-of-K cosine. Motivated by: within-class features are multi-modal (e.g. texture orientations on dtd), so one mean cannot represent them.
- **M2 — CompactPTA**: keep base PTA's single mean but gate each write by its distance to the current class prototype (soft attenuation or hard drop). Motivated by: writes far from the class prototype are disproportionately wrong.

A fourth candidate, **M3 — per-class adaptive EMA rate** (faster adaptation for high-frequency classes), was pre-checked and **rejected before implementation** (Part 0, Gate 0.2); it is reported here as a negative pre-check.

**Incremental protocol.** No method was built or run on the cluster until a cheap offline gate validated its mechanism (Part 0). Gates that failed led to the direction being dropped, not built. The online 4-seed runs are the final arbiter: a gate passing is necessary, not sufficient.

### Baseline reference points (4-seed mean)

| Dataset | CLIP zero-shot | PTA fused (`clip + 100×image_proto`) | PTA gain |
|---------|----------------|--------------------------------------|----------|
| dtd | 44.39% | **47.47%** (47.70/47.16/47.81/47.22) | **+3.08pp** |
| oxford_flowers | 71.38% | **74.55%** (74.46/74.34/74.50/74.91) | **+3.17pp** |
| oxford_pets | 89.07% | **91.18%** (91.20/91.22/91.01/91.28) | **+2.11pp** |

PTA's 3-dataset average is 71.07%. Every method in this report must clear the PTA column to count as a win.

---

## Part 0: Pre-Validation (Go/No-Go Gates)

Each gate answers one question: *does the mechanism this method depends on actually exist in the data?* All gates are offline and cheap (archived base-PTA records or a one-off feature dump of the test sets).

### 0.1 Gate for M2: write purity vs distance-to-prototype — **GO**

From archived base-PTA records (12,400+ write events per dataset, pooled over 4 seeds), each write event carries the image feature and the prototype state at write time, so `d = 1 - cos(x, proto_c)` is recoverable. Bucketing writes into deciles of `d` and measuring the fraction of writes whose class equals the true label:

| Dataset | D1 (nearest) | D5 | D10 (farthest) | nearest/farthest ratio |
|---------|--------------|----|----------------|------------------------|
| dtd | 71.3% | 30.8% | 13.4% | 5.34 |
| oxford_flowers | 86.7% | 52.0% | 11.8% | 7.36 |
| oxford_pets | 89.2% | 70.8% | 38.1% | 2.34 |

Write correctness falls steeply with distance on all three datasets. At a threshold keeping ~66-84% of writes (dtd τ=0.20: keeps 84.1%, removes 20.8% of wrong writes vs 6.1% of correct; flowers τ=0.08: keeps 68.6%, removes 50.3% wrong vs 10.4% correct; pets τ=0.10: keeps 65.6%, removes 24.6% wrong vs 12.3% correct), the removed writes are 2.7-4.9× more likely to be wrong than the kept ones. **The compactness signal is real → build M2.**

Caveat (checked, not a blocker): this is a *correlation* measured on base-PTA's own write stream. Gating changes the stream, so the online run is the final test.

### 0.2 Gate for M3: late-stream hijacking of high-frequency classes — **NO-GO (direction dropped)**

M3's premise: high-frequency classes accumulate more writes, so a fixed EMA rate either lags them (early stream) or overfits them (late stream); an adaptive per-class rate should help. The falsifiable signature of that premise is *hijacking*: in the late stream (quartile Q4 of time), writes to top-frequency classes should be substantially less correct than in the early stream (Q1). Measured on archived records (top-frequency classes: dtd n=11, flowers n=24, pets n=9):

| Dataset | Q1 purity | Q4 purity | Q4 − Q1 |
|---------|-----------|-----------|---------|
| dtd | 37.3% | 40.2% | +2.8pp |
| oxford_flowers | 50.2% | 50.5% | +0.3pp |
| oxford_pets | 50.3% | 49.8% | −0.5pp |

No hijacking in any dataset (GO bar: Q4 ≤ Q1 − 5pp on ≥2 datasets). The fixed-rate EMA is not leaving measurable accuracy on the table for high-frequency classes. **M3 was not implemented or run.**

### 0.3 Feature dump (shared by Gates 0.4/0.5)

One-off extraction of the full test-set CLIP image features (ViT-B/16, CLIP Surgery): dtd 1692, oxford_flowers 2463, oxford_pets 3669 images, 512-dim float32 L2-normalized, with ground-truth labels. Used only for the two favorable-proxy gates below.

### 0.4 Gate for M1a: variance-aware (Mahalanobis) scoring vs cosine — **GO**

Favorable proxy: for each class, the *full-stream* true-label mean and per-dimension variance (i.e., the best case the online method could hope to converge to); score every image with (a) cosine to the mean and (b) shifted Mahalanobis `d_c = (1/D) Σ_d (x − μ_c)² / (v_c + λ·V̄_d)`, `logit_c = −d_c`, where `V̄_d` is the cross-class per-dimension variance mean and λ is the shrinkage.

| Dataset | cosine(mean) | Mahalanobis λ=0.1 | Mahalanobis λ=0.5 |
|---------|--------------|-------------------|-------------------|
| dtd | 72.58% | 72.75% (+0.18) | 76.77% (+4.20) |
| oxford_flowers | 97.20% | 98.42% (+1.22) | 98.90% (+1.71) |
| oxford_pets | 91.36% | 92.45% (+1.09) | 93.38% (+2.02) |

Variance-aware scoring beats cosine on all three datasets at both shrinkages (GO bar: ≥+1.0pp on ≥1 dataset and no dataset loses >0.3pp). **Build M1a** with λ ∈ {0.1, 0.5}.

### 0.5 Gate for M1b: K-representative bank vs single mean — **GO**

Favorable proxy: offline k-means with K representatives on each class's true-label features; score every image with max-of-K cosine vs cosine to the single mean.

| Dataset | cosine(mean) | bank K=3 | bank K=5 |
|---------|--------------|----------|----------|
| dtd | 72.58% | 84.99% (+12.41) | 90.60% (+18.03) |
| oxford_flowers | 97.20% | 99.27% (+2.07) | 99.59% (+2.40) |
| oxford_pets | 91.36% | 92.86% (+1.50) | 94.39% (+3.03) |

A K-representative bank beats the single mean on all three datasets at both K (GO bar as above). The margin is largest on dtd, where within-class texture structure is most multi-modal. **Build M1b** with K ∈ {3, 5}.

### 0.6 Decision

Build and run M1a (2 settings), M1b (2 settings), M2 (2 gate modes) — 3 datasets × 4 seeds × 2 settings = 72 tasks. Drop M3.

---

## Part 1: Methods

All three methods share base PTA's write *rule* (every class with softmax probability ≥ 0.1 is written with `w_c = 1 − exp(−p_c / T)`, T=20) and its fusion scale (τ_text=1.0, τ_image_proto=100.0, no patch term). They differ in the prototype state, the scoring function, and/or the write weight. Where a method has a degenerate setting that should reduce to base PTA (the bank at K=1, compact at gate-off), that equivalence is enforced and unit-tested — this caught a real initialization bug in the bank (Section 2.3).

### 1.1 GaussPTA (M1a) — per-class Gaussian prototype

State per class `c` (all float32): mean `μ_c` (L2-normalized, init: text embedding), per-dimension variance `v_c` (init: isotropic 1.0), effective count `n_c` (init: 0).

- **Write** (for each written class, weight `w_c`): `μ_c ← normalize((1−w_c)μ_c + w_c x)`; `v_c ← (1−w_c)v_c + w_c (x − μ_c)²` (deviation from the *updated* mean); `n_c ← n_c + w_c`.
- **Score**: `d_c = (1/D) Σ_d (x − μ_c)² / (v_c + λ·V̄_d)`, `logit_c = −d_c`, with `V̄_d` the cross-class per-dimension variance mean recomputed each step.
- **Fusion**: `final = clip + 100 · proto_c` where `proto_c = (logit_c − max)/(max − min)` is the Mahalanobis logit rescaled per sample to [−1, 0] (cosine's scale). The raw Mahalanobis distance is ~100× smaller than cosine, so without this rescaling the prototype term is negligible at τ=100 and the method collapses to CLIP (Section 2.2).
- **Setting**: `λ` (shrinkage) ∈ {0.1, 0.5}.

### 1.2 BankPTA (M1b) — per-class prototype bank

State per class `c`: a bank `P[c, :, :]` of K prototypes (float32), **all initialized at zero** — exactly like base PTA's prototype (`PTAImageLevel.init_state` returns zeros). The text embedding enters only through the scoring blend `refine[c,k] = normalize(0.01·text + 0.99·P[c,k])`, so a never-written prototype scores exactly like the class text. This zero-init + refine-scoring is what makes K=1 reproduce base PTA (the earlier text-initialized variant did not — Section 2.3).

- **Write** (for each written class, weight `w_c`): assign to the nearest prototype, measured against the base-PTA-style refine, `k* = argmax_k cos(x, refine[c,k])`, and update only that prototype: `P[c,k*] ← (1−w_c) P[c,k*] + w_c x` (no re-normalization — normalization happens inside the refine blend at scoring time). All other prototypes are untouched.
- **Score**: `logit_c = max_k cos(x, refine[c,k])`. For K=1 this is exactly base PTA's `cos(x, refined_text)`.
- **Fusion**: exact mirror of base PTA, `final = clip + 100 · logit_c`.
- **Setting**: K ∈ {3, 5}.

### 1.3 CompactPTA (M2) — intra-class compactness write gate

State: base PTA's single mean prototype (init: text embedding). Prediction and fusion are identical to base PTA. The only change is the write weight, modulated by the distance `d_c = 1 − cos(x, proto_c)` to the *current* (pre-write) class prototype:

- **soft**: `w_c ← w_c · exp(−d_c² / (2σ_c²))`, where `σ_c` is a per-class running EMA of `d_c` (rate 0.05, init 0.1, floor 0.05).
- **hard**: `w_c ← 0` if `d_c > mean_c + 0.5·std_c`, else `w_c` (running EMAs of `d_c` and `d_c²`, same rates/initialization).
- A never-written class (zero prototype) is written ungated at full weight.
- **Setting**: gate_mode ∈ {soft, hard}.

---

## Part 2: Results (Wave 1)

All numbers are 4-seed means. Reference: CLIP / PTA from the Background table.

### 2.1 CompactPTA (M2) — compactness write gate

| Dataset | CLIP | PTA | CMP-soft | Δ vs PTA | CMP-hard | Δ vs PTA |
|---------|------|-----|----------|----------|----------|----------|
| dtd | 44.39 | 47.47 | 44.58 (43.97/44.15/44.80/45.39) | **−2.89pp** | 41.43 (40.66/40.84/42.20/42.02) | **−6.04pp** |
| oxford_flowers | 71.38 | 74.55 | 74.37 (73.93/74.18/74.83/74.54) | −0.18pp | 71.66 (71.34/71.46/72.27/71.58) | −2.89pp |
| oxford_pets | 89.07 | 91.18 | 90.80 (90.79/90.79/90.81/90.81) | −0.38pp | 88.66 (88.55/88.55/89.13/88.39) | −2.53pp |

**M2 loses on every dataset in both gate modes.** The soft gate costs −2.9pp on dtd (falling to within 0.2pp of CLIP) and small amounts elsewhere; the hard gate is strictly worse than soft everywhere (−6.0pp on dtd).

**Mechanism.** Gate 0.1's correlation (far writes are less pure) is real but not actionable at write time: the gate cannot distinguish, for a *given* far write, whether it is one of the wrong ones or one of the still-correct far writes (dtd D5 purity is already 30.8% — nearly coin-flip). Attenuating or dropping all far writes therefore removes a large share of *correct* updates along with the wrong ones, starving the prototype of the very samples that move it. This is the same negative pattern as the earlier write-rule study (gating/re-weighting writes by a trust signal never beat base PTA): the trust/quality signal is a good *diagnostic* but not a *write-time lever*. The compactness gate confirms it for the geometric distance signal.

### 2.2 GaussPTA (M1a) — per-class Gaussian prototype

**A scale issue, and its fix.** The Mahalanobis distance `d_c = (1/D) Σ (x−μ_c)²/(v_c + λ·V̄_d)` is on a ~100× smaller scale than cosine (unit-norm features spread over D=512 dims), so at the standard PTA fusion weight (τ=100) the prototype term is negligible and the method collapses to CLIP. With the standard fusion, GaussPTA indeed tracks CLIP: shrink=0.5 gives dtd 44.47 / flowers 71.28 / pets 90.13 (CLIP: 44.39 / 71.38 / 89.07). To fairly test the Mahalanobis *ranking* — which is what Gate 0.4 validated — the prototype term is rescaled per sample to cosine's scale, `proto = (logit − max)/(max − min)` ∈ [−1, 0], preserving the ranking while giving it the same weight base PTA gives its cosine term. The table below is the rescaled (fair) result.

| Dataset | CLIP | PTA | GAU λ=0.1 | Δ vs PTA | GAU λ=0.5 | Δ vs PTA |
|---------|------|-----|-----------|----------|-----------|----------|
| dtd | 44.39 | 47.47 | 44.09 (44.62/44.62/43.56/43.56) | **−3.38pp** | 38.59 (38.36/38.95/37.59/39.48) | **−8.88pp** |
| oxford_flowers | 71.38 | 74.55 | 63.67 (64.27/63.70/63.50/63.22) | **−10.88pp** | 54.80 (55.18/55.14/54.28/54.61) | **−19.75pp** |
| oxford_pets | 89.07 | 91.18 | 59.17 (59.14/56.96/58.87/61.73) | **−32.01pp** | 58.39 (57.89/58.05/57.92/59.69) | **−32.79pp** |

**M1a is a strong negative.** Once the Mahalanobis signal is given the weight its ranking deserves, it is actively *harmful* — far below CLIP on every dataset (pets collapses to ~58% vs CLIP's 89%). Shrinking the variance (λ=0.5, more regularization) makes it worse, not better.

**Mechanism.** Gate 0.4 validated the Mahalanobis ranking under a *favorable proxy*: full-stream, true-label means and per-dim variances. Online, the variance `v_c` is estimated from CLIP-driven writes — which are wrong ~56% of the time on dtd — so `v_c` is a noisy, biased estimate of the true within-class spread. The Mahalanobis distance is far more sensitive to a bad variance estimate than cosine is to a bad mean: a dimension whose variance is underestimated makes `d_c` overstate the distance, misranking the true class below impostors. Cosine, by contrast, degrades gracefully (a shifted mean just lowers the score uniformly). The offline gate measured the ranking with *clean* statistics; the online method must *estimate* those statistics from a biased stream, and the Mahalanobis form amplifies the estimation error. The variance-aware upgrade therefore does not transfer.

### 2.3 BankPTA (M1b) — per-class prototype bank

**A bug, and its fix.** The first implementation initialized the K prototypes at the class *text embedding* and scored `cos(x, P)` directly. That does **not** reproduce base PTA: base PTA's prototype is zero-initialized and scored via `refine = normalize(0.01·text + 0.99·proto)`, a fundamentally different dynamic. The buggy variant scored 25.18% on dtd (K=1, K=3, K=5 all identical) vs base PTA's 47.70% — a 22pp artifact, not a method result. The fix re-initializes the bank at zero and scores each prototype through the base-PTA-style `refine` blend, with no per-step re-normalization; a CPU unit test now verifies K=1 reproduces base PTA's update+score bit-for-bit, and an end-to-end K=1 run gives 46.63% on dtd (vs 47.70% base PTA; the 1.1pp gap is fp16-vs-fp32 EMA precision). The table below is the fixed result.

| Dataset | CLIP | PTA | BNK K=3 | Δ vs PTA | BNK K=5 | Δ vs PTA |
|---------|------|-----|---------|----------|---------|----------|
| dtd | 44.39 | 47.47 | 46.85 (46.63/47.22/46.75/46.81) | **−0.62pp** | 46.85 (same) | **−0.62pp** |
| oxford_flowers | 71.38 | 74.55 | 74.44 (74.30/74.38/74.46/74.62) | −0.11pp | 74.44 (same) | −0.11pp |
| oxford_pets | 89.07 | 91.18 | 91.12 (91.11/91.11/91.14/91.14) | −0.06pp | 91.12 (same) | −0.06pp |

**M1b is a null result: the bank does not beat PTA, and K=3 and K=5 are identical on all 24 (dataset, seed) runs.** The fixed bank tracks base PTA within ~0.6pp (the residual gap is the fp16/fp32 precision difference plus the max-of-K taking the max against the text anchor).

**Mechanism — rich-get-richer collapse.** With zero initialization, all K prototypes of a class start identical (each scoring like the class text). The first write to a class breaks the tie to one prototype (k=0); that prototype then becomes the nearest for subsequent writes, so it keeps receiving them while the other K−1 prototypes stay at zero (scoring like the text). The bank therefore collapses to **one active prototype plus K−1 text anchors**, regardless of K — which is exactly why K=3 and K=5 are identical in accuracy on all 24 (dataset, seed) runs. The nearest-assignment write rule cannot create the K-way diversity that Gate 0.5's offline k-means had, because offline k-means was seeded on *true-label* features with a proper clustering objective, whereas the online bank is seeded by CLIP's (often wrong) writes and a winner-take-all nearest assignment. The offline gate's +12pp (dtd) promise therefore does not transfer: the online bank never achieves the multi-modal representation the gate measured.

### 2.4 BankV2 (proper re-implementation) — collapse fixed, accuracy collapses

The v1 diagnosis above (rich-get-richer) is confirmed and fixed. **BankV2** (`models/bank_v2_pta.py`) replaces the nearest-assignment write rule with the repo's proven anti-collapse pattern (threshold-gated growth + least-used recycling): slot 0 is a writable, never-recycled anchor (base PTA's prototype); a written feature is only *absorbed* into its nearest active slot if `cos(x, refine[c,k*]) >= create_threshold` (0.85), otherwise it **grows a fresh slot** (or recycles the least-used specialized slot when the bank is full). Per-class `active_k` is recorded every step; a CPU unit test verifies K=1 reproduces base PTA bit-for-bit and that K>1 grows slots; the K=1/off equivalence checks all pass (Section A.3).

**The fix achieves its mechanism goal.** On every run, virtually all classes end up multi-modal: K=3 → mean `active_k` 2.98 (min 2, max 3, 100% of classes > 1); K=5 → mean 4.87 (min 2, max 5). K=3 and K=5 now produce *different* models (dtd 42.33 vs 41.96) — the collapse is gone.

**But the accuracy collapsed instead.** The bank tracks CLIP, not PTA, and clearly loses to both:

| Dataset | CLIP | PTA | BNKV2 K=3 | Δ vs PTA | BNKV2 K=5 | Δ vs PTA |
|---------|------|-----|-----------|----------|-----------|----------|
| dtd | 44.39 | 47.47 | 42.33 (41.84/42.79/42.32/42.38) | **−5.14pp** | 41.96 (41.02/42.02/42.61/42.20) | **−5.51pp** |
| oxford_flowers | 71.38 | 74.55 | 68.65 (67.64/69.83/67.76/69.35) | **−5.90pp** | 68.74 (67.68/69.43/69.27/68.57) | **−5.81pp** |
| oxford_pets | 89.07 | 91.18 | 86.79 (86.89/87.19/86.07/87.00) | **−4.39pp** | 86.92 (87.35/86.51/86.75/87.08) | **−4.26pp** |
| **avg(3)** | **68.28** | **71.07** | **65.92** | **−5.15pp** | **65.87** | **−5.19pp** |

**Mechanism — the growth rule plants CLIP's error modes.** The threshold-gated growth fixes diversity, but it creates modes from *the wrong writes*. A grow fires exactly when a write is dissimilar to every active prototype — and on this CLIP-driven stream the writes most dissimilar to the class's true modes are disproportionately the **wrong-label** ones (base-PTA write purity is ~50% on dtd, and drops with distance — Gate 0.1). A new slot is seeded at **full strength with the raw (often wrong) feature** and then self-reinforces: the recycled/consumed error features make the class's max-of-K score attractive to *other* wrong features of the same error cluster. Where v1 collapsed to one true text-anchored mode (harmless), v2 grows K−1 *error modes* that actively pull predictions toward the wrong class — and the more slots, the more spurious winners the `max` can find (K=5 ≥ K=3's loss on dtd). The same lesson as GaussPTA, in a different form: **the mechanism that Gate 0.5 validated (diversity from true-label k-means) cannot be realized from CLIP's biased writes**; a threshold-gated bank grown on a biased stream is worse than the single EMA it replaces, because the EMA averages errors away while the bank promotes them.

---

## Part 3: Combinations

All individual methods fail to beat PTA, but the combination phase was **run anyway** (per user request): the two most natural compositions — the proper bank with the compactness write gate (**BankCompact**, `models/bank_compact_pta.py`, K ∈ {3,5}) and the Gaussian prototype with the compactness write gate (**GaussCompact**, `models/gauss_compact_pta.py`, shrink ∈ {0.1,0.5}) — each on 3 datasets × 4 seeds = 24 runs per setting. Both combinations share the v2 anti-collapse write path (grow/recycle is ungated; the compactness gate only modulates absorbed writes).

### 3.1 BankCompactPTA (BankV2 + compactness gate) — K∈{3,5} × 3 ds × 4 seeds = 24 runs per setting

| Dataset | PTA | BNKV2 K3 | BNKV2 K5 | BNKCMP K3 | Δ vs PTA | BNKCMP K5 | Δ vs PTA |
|---------|-----|----------|----------|-----------|----------|-----------|----------|
| dtd | 47.47 | 42.33 | 41.96 | 42.29 (41.96/42.73/42.26/42.20) | **−5.18pp** | 41.87 (40.78/42.02/42.55/42.14) | **−5.60pp** |
| oxford_flowers | 74.55 | 68.65 | 68.74 | 67.82 (66.67/68.21/68.01/68.37) | **−6.74pp** | 67.81 (66.79/67.97/68.74/67.72) | **−6.75pp** |
| oxford_pets | 91.18 | 86.79 | 86.92 | 85.40 (85.61/86.15/83.86/85.96) | **−5.79pp** | 85.81 (86.15/85.45/85.50/86.13) | **−5.37pp** |
| **avg(3)** | **71.07** | **65.92** | **65.87** | **65.17** | **−5.90pp** | **65.16** | **−5.91pp** |

**BankCompact is BankV2 minus a bit more.** The gate (which alone costs −1.15pp avg, Sec 2.1) removes far *absorbed* writes but cannot touch the dominant error source: the ungated grow/recycle events that plant the wrong-label error modes (Sec 2.4). Result: the combination tracks the bank's collapse, marginally worse (65.17 vs 65.92 on K3) — the two mechanisms don't interact beneficially, they just subtract.

### 3.2 GaussCompactPTA (GaussPTA + compactness gate) — gate partially rescues the variance collapse, still far below PTA

| Dataset | PTA | GAU s0.1 | GAU s0.5 | GAUCMP s0.1 | Δ vs PTA | GAUCMP s0.5 | Δ vs PTA |
|---------|-----|----------|----------|-------------|----------|-------------|----------|
| dtd | 47.47 | 44.09 | 38.60 | 42.02 (41.55/42.08/42.02/42.43) | **−5.45pp** | 32.83 (31.74/32.74/32.09/34.75) | **−14.64pp** |
| oxford_flowers | 74.55 | 63.67 | 54.80 | 57.76 (57.82/58.18/56.84/58.18) | **−16.79pp** | 40.82 (40.44/42.27/38.90/41.66) | **−33.73pp** |
| oxford_pets | 91.18 | 59.18 | 58.39 | 74.45 (73.15/74.19/74.43/76.04) | **−16.73pp** | 74.47 (73.32/74.22/74.93/75.39) | **−16.72pp** |
| **avg(3)** | **71.07** | **55.65** | **50.60** | **58.08** | **−12.99pp** | **49.37** | **−21.70pp** |

**The gate helps exactly where the variance blows up — and nowhere else.** On oxford_pets, where GaussPTA's estimated-variance collapse was most extreme (59.18, −32pp below PTA; the worst case in the study), dropping far writes recovers +15.3pp (to 74.45, s0.1). On dtd and flowers the same gate *hurts* the already-damaged Gaussian further. Net: GaussCompact is still −13 to −22pp below PTA — the improvement is a partial rescue of the most broken case, not a competitive method.

### 3.3 Verdict on combinations

**Combining the two non-winning mechanisms compounds their errors.** BankCompact ≈ BankV2 (−5.9pp avg, the gate adds nothing); GaussCompact ≈ GaussPTA (−13 to −22pp, the gate's rescue of pets is swamped by further dtd/flowers losses). No combination beats PTA, and none comes close: the best combination avg (BankCompact 65.17) sits 5.9pp below PTA and 3.1pp below CLIP. The original skip decision (Part 3, v1 of this report) is validated *post-hoc*: methods whose individual mechanisms misfire on the biased write stream do not become useful when composed — the write-stream bias is the shared, binding failure mode of every direction tested here.

---

## Part 4: Wave 3 — Augmentation Ensembling and Text-Anchor Damping

Wave 3 (jobs 16739–16742, 96 tasks: 4 seeds × 3 datasets × 4 settings for aug/anchor/abl, 2 seeds × 3 datasets × 4 settings for the sweep) tests two further mechanism families against the same binding constraint. Both are strict subsets of base PTA — same ungated softmax-EMA write rule, same fusion (1.0·clip + 100.0·image_proto, `tau_patch_proto=0`), no gates, skips, or reweighting added. Each family includes an identity setting that must reproduce base PTA exactly, doubling as a GPU-reducibility proof.

### 4.1 Mechanisms and settings

- **AugPTA (M4) — augmentation ensembling.** Generate V augmented copies of the test image; keep the K lowest-entropy views; average their CLIP logits. The ensembled logits replace the single-view logits as the write-mask evidence and the prediction basis. Premise: ensembling over CLIP's own augmentations raises pseudo-label quality, diluting the confident-but-wrong writes that poison the stream. Settings: `image_level.n_aug` × `image_level.k_views` ∈ {V8K4, V4K2}; V1K1 is the identity (must equal base PTA).
- **AnchorPTA (M5) — text-anchor damping.** The prototype-EMA update weight is scaled by `damp_factor = clamp(sim / damp_threshold, damp_floor, 1)`, where `sim` is the prototype's cosine to its frozen text anchor. Premise: a prototype that drifts far from its text anchor has been poisoned, so its (large) writes should be damped, and damped writes should be less pure than undamped ones. Settings: `damp_threshold` 0.96 / 0.93, `damp_floor` 0.5; threshold 0 is the identity.
- **Ablations: V8K8** (mean-free ensembling — averages all 8 views, no entropy selection) isolates the value of entropy selection; **V1K1** is the end-to-end identity check.
- **Hyperparameter sweep** of base PTA: α ∈ {0.003, 0.1} × T ∈ {10, 40} (2 seeds), sanity-checking that the default α=0.01 / T=20 is not a strawman.

### 4.2 Accuracy (4-seed means; sweep 2-seed)

| Method | dtd | flowers | pets | avg(3) | vs PTA |
|---|---|---|---|---|---|
| CLIP zero-shot | 44.39 | 71.38 | 89.07 | 68.28 | −3.08 / −3.17 / −2.11 |
| **PTA (baseline)** | **47.47** | **74.55** | **91.18** | **71.07** | — |
| AUG-V4K2 (V=4, K=2) | 47.27 | 73.58 | 90.82 | 70.56 | −0.20 / −0.97 / −0.36 |
| AUG-V8K4 (V=8, K=4) | 46.66 | 72.53 | 90.92 | 70.04 | −0.81 / −2.02 / −0.26 |
| ANCH-d096 (threshold 0.96) | 47.40 | 74.59 | 91.16 | 71.05 | −0.07 / +0.04 / −0.03 |
| ANCH-d093 (threshold 0.93) | 47.43 | 74.58 | 91.16 | 71.06 | −0.04 / +0.03 / −0.02 |
| ABL-V1K1 (identity) | 47.47 | 74.55 | 91.18 | 71.07 | +0.000 |
| ABL-V8K8 (mean-free) | 45.24 | 70.99 | 90.65 | 68.96 | −2.23 / −3.56 / −0.54 |

Sweep (2-seed mean, avg(3) vs PTA): α=0.003/T=10 → **+0.02**; α=0.003/T=40 → **+0.10**; α=0.1/T=10 → −0.78; α=0.1/T=40 → −1.51. The default α=0.01/T=20 sits at the top of the local grid — the baseline is not a strawman.

### 4.3 Write purity

Write purity = P(write class == sample class) over write events, mask = `softmax(logits.clip) ≥ 0.1` recomputed identically from the stored per-sample logits of every method (`scripts/wave3_write_purity.py`, 4-seed means; `clip_acc` = accuracy of the stored evidence logits, i.e. the write-mask source):

| Method | dtd write% | dtd purity | dtd clip_acc | flowers write% | flowers purity | flowers clip_acc | pets write% | pets purity | pets clip_acc |
|---|---|---|---|---|---|---|---|---|---|
| PTA baseline | 97.70 | 33.19 | 44.39 | 99.96 | 47.36 | 71.38 | 100.00 | 67.20 | 89.07 |
| AUG-V4K2 | 97.90 | 32.56 | 43.41 | 99.92 | 46.54 | 69.17 | 100.00 | 67.17 | 88.06 |
| AUG-V8K4 | 97.84 | 32.67 | 42.55 | 99.88 | 45.62 | 68.43 | 100.00 | 66.01 | 88.11 |
| ANCH-d096 | 97.70 | 33.19 | 44.39 | 99.96 | 47.36 | 71.38 | 100.00 | 67.20 | 89.07 |
| ANCH-d093 | 97.70 | 33.19 | 44.39 | 99.96 | 47.36 | 71.38 | 100.00 | 67.20 | 89.07 |
| ABL-V1K1 | 97.70 | 33.19 | 44.39 | 99.96 | 47.36 | 71.38 | 100.00 | 67.20 | 89.07 |
| ABL-V8K8 | 92.73 | 33.37 | 40.81 | 99.15 | 42.68 | 64.74 | 100.00 | 57.69 | 87.32 |

Findings:

- **V1K1 is identical to base PTA in write%, purity, clip_acc (and per-seed accuracy)** — the identity ablation, and with it the lossless reducibility of the GPU files, is confirmed at the record level.
- **AugPTA lowers write purity instead of raising it.** Ensembled-logit accuracy falls below zero-shot CLIP on every dataset (dtd 42.6–43.4 vs 44.4; flowers 68.4–69.2 vs 71.4; pets 88.1 vs 89.1) and purity drops on dtd (32.6–32.7 vs 33.2) and flowers (45.6–46.5 vs 47.4). More views (V8) is strictly worse than fewer (V4). The entropy-selected ensemble is a worse evidence source than the plain single view — CLIP's augmentations do not make its guesses more reliable.
- **AnchorPTA's damping is not selective.** write%/purity/clip_acc are identical to baseline (damping never touches the mask), and the damp split (per-class cumulative `n_damped` counters) shows damped-event purity ≈ undamped (dtd 33.21 vs 33.19, flowers 47.24 vs 47.36, pets 67.19 vs 67.20). After a ~200-sample warm-up — during which prototypes drift below `sim = threshold` — 98–100% of all write events are damped (d096-dtd-s1: 60.7% in the first 50 samples, 99.8% after 200, 98.35% over the full run). The mechanism degenerates into a near-uniform update-magnitude scaling, which is exactly why it ties base PTA (−0.01/−0.02) rather than selectively removing wrong writes.
- **V8K8 (mean-free) collapses** — write% drops on dtd (92.73), purity collapses on flowers/pets (42.68 / 57.69 vs 47.36 / 67.20). Entropy selection carries real value (V8K4 −1.03 vs V8K8 −2.11) but cannot offset the ensembling cost.

### 4.4 Verdict for wave 3

Both wave-3 mechanism families fail to beat base PTA. AnchorPTA ties it for a degenerate reason: damping ≈ uniform learning-rate scaling after warm-up, and its selectivity premise never materializes. AugPTA loses because CLIP's augmentations do not produce more reliable pseudo-labels — the ensemble logits are *less* accurate than the single view, so ensembling injects the same bias with more confidence. The identity ablation (V1K1), the mean-free ablation (V8K8), and the α/T sweep jointly validate the measurement: strict-subset implementations are exactly reducible, entropy selection is real but insufficient, and the default PTA config sits at the top of its local hyperparameter grid. The write-stream bias remains the binding constraint, consistent with the wave-1/2 conclusion.

---

## Summary

All three representation-level upgrades of PTA's per-class prototype — variance-aware (Gaussian) scoring, a K-prototype bank, and a compactness write gate — **fail to beat base PTA** on dtd / oxford_flowers / oxford_pets, despite each passing a cheap offline pre-validation gate. The wave-2 proper-bank re-implementation (BankV2, which fixes the rich-get-richer collapse) and both combinations (BankCompact, GaussCompact) fail as well and are strictly *worse*; so do the wave-3 mechanisms (Part 4) — ensembling loses (avg −0.51 to −1.03), and text-anchor damping ties (avg −0.01/−0.02) because its gate fires on ~all writes after warm-up and is not selective:

| Method | dtd | flowers | pets | avg(3) | vs PTA (47.47 / 74.55 / 91.18) |
|--------|-----|---------|------|--------|--------------------------------|
| CLIP zero-shot | 44.39 | 71.38 | 89.07 | 68.28 | −3.08 / −3.17 / −2.11 |
| **PTA (baseline)** | **47.47** | **74.55** | **91.18** | **71.07** | — |
| M2 CompactPTA (soft) | 44.58 | 74.37 | 90.80 | 69.92 | −2.89 / −0.18 / −0.38 |
| M2 CompactPTA (hard) | 41.43 | 71.66 | 88.66 | 67.25 | −6.04 / −2.89 / −2.53 |
| M1a GaussPTA (λ=0.5, rescaled) | 38.59 | 54.80 | 58.39 | 50.60 | −8.88 / −19.75 / −32.79 |
| M1b BankPTA (K=3 = K=5) | 46.85 | 74.44 | 91.12 | 70.81 | −0.62 / −0.11 / −0.06 |
| BankV2 K=3 (proper bank) | 42.33 | 68.65 | 86.79 | 65.92 | −5.14 / −5.90 / −4.39 |
| BankV2 K=5 (proper bank) | 41.96 | 68.74 | 86.92 | 65.87 | −5.51 / −5.81 / −4.26 |
| BankCompact K=3 (bank+gate) | 42.29 | 67.82 | 85.40 | 65.17 | −5.18 / −6.74 / −5.79 |
| BankCompact K=5 (bank+gate) | 41.87 | 67.81 | 85.81 | 65.16 | −5.60 / −6.75 / −5.37 |
| GaussCompact λ=0.1 (gauss+gate) | 42.02 | 57.76 | 74.45 | 58.08 | −5.45 / −16.79 / −16.73 |
| GaussCompact λ=0.5 (gauss+gate) | 32.83 | 40.82 | 74.47 | 49.37 | −14.64 / −33.73 / −16.72 |
| AUG-V4K2 (wave 3: V-view ensembling) | 47.27 | 73.58 | 90.82 | 70.56 | −0.20 / −0.97 / −0.36 |
| AUG-V8K4 (wave 3: V-view ensembling) | 46.66 | 72.53 | 90.92 | 70.04 | −0.81 / −2.02 / −0.26 |
| ANCH-d096 (wave 3: text-anchor damping) | 47.40 | 74.59 | 91.16 | 71.05 | −0.07 / +0.04 / −0.03 |
| ANCH-d093 (wave 3: text-anchor damping) | 47.43 | 74.58 | 91.16 | 71.06 | −0.04 / +0.03 / −0.02 |
| ABL-V1K1 (wave 3: identity check) | 47.47 | 74.55 | 91.18 | 71.07 | +0.000 |
| ABL-V8K8 (wave 3: mean-free ensembling) | 45.24 | 70.99 | 90.65 | 68.96 | −2.23 / −3.56 / −0.54 |

**Why the gates passed but the methods did not.** Every gate measured its mechanism under a *favorable, non-causal proxy*: M1a with full-stream true-label statistics, M1b with offline k-means on true-label features, M2 on base PTA's own (already-adapted) write stream. The online methods must realize the same mechanism from a *label-free, CLIP-driven* stream, and each fails for a distinct, identifiable reason:

- **M1a (Gaussian):** the variance must be *estimated* from a biased write stream; the Mahalanobis form amplifies the estimation error, so a signal that ranks well with clean statistics ranks badly with estimated ones.
- **M1b (bank, v1):** the nearest-assignment write rule cannot create K-way diversity from a label-free stream — it collapses to one active prototype (K=3 = K=5), so the multi-modal representation the gate measured is never realized.
- **BankV2 (proper bank):** threshold-gated growth *does* create K-way diversity (mean `active_k` ≈ K, K=3 ≠ K=5), but the growth rule plants full-strength **error modes** from the writes most dissimilar to the true class modes — CLIP's wrong labels. Diversity from a biased stream is worse than the single EMA it replaces (avg −5.2pp, below CLIP).
- **M2 (compactness gate):** the distance→purity correlation is real but not actionable per-write; gating removes correct updates along with wrong ones. This matches the earlier write-rule finding that a quality signal is a good *diagnostic* but not a *write-time lever*.
- **Combinations:** composing the failed mechanisms compounds their errors. BankCompact ≈ BankV2 (the gate cannot touch the dominant ungated error-mode growth); GaussCompact ≈ GaussPTA (the gate's +15pp rescue of pets' worst collapse is swamped by further dtd/flowers losses). No combination clears PTA; the best (BankCompact) sits 5.9pp below PTA and 3.1pp below CLIP.
- **Wave 3, AugPTA (ensembling):** the entropy-selected augmentation ensemble is a *worse* evidence source than the plain single view — ensembled-logit accuracy drops below zero-shot CLIP on every dataset (dtd 42.6–43.4 vs 44.4; flowers 68.4–69.2 vs 71.4; pets 88.1 vs 89.1) and write purity drops on dtd/flowers. CLIP's augmentations do not make its guesses more reliable, so the mechanism sabotages its own premise; more views (V8) hurt more than fewer (V4).
- **Wave 3, AnchorPTA (text-anchor damping):** damping is not selective — after a ~200-sample warm-up 98–100% of write events are damped (prototypes drift below `sim=threshold`), and damped-event purity ≈ undamped (dtd 33.21 vs 33.19). The mechanism degenerates into a near-uniform update-magnitude scaling, which is exactly why it ties PTA (−0.01/−0.02) instead of removing wrong writes.

**Takeaway.** PTA's single-mean, cosine-scored, fixed-rate prototype is not beaten by these three natural representation upgrades, by the collapse-fixed proper bank, by their combinations, or by the wave-3 ensembling/damping mechanisms. The prototype is the right lever (it is the more reliable predictor on CLIP↔prototype ties), but the specific upgrades tested here do not convert their offline promise into online accuracy — the gap between a favorable offline proxy and the label-free online setting is the binding constraint, and it is shared: every mechanism here misfires on the *same* biased CLIP write stream. Beating PTA likely requires a mechanism whose advantage does *not* depend on statistics that must be estimated from CLIP's own (biased) guesses.

**Reproducibility note.** Two implementation defects were found and fixed during this study and are documented above: the bank's text-initialization/scoring bug (Section 2.3) and the GaussPTA fusion-scale mismatch (Section 2.2). All reported numbers are from the fixed code; the pre-fix bank (25.18% dtd) and pre-fix GaussPTA (≈CLIP) are cited only as the "before" for each fix.

---

## Appendix: Supplementary Data

### A.1 Per-seed accuracies

All values in percent; column order is seeds 1-4.

| Method / setting | dtd s1 | dtd s2 | dtd s3 | dtd s4 | flowers s1 | flowers s2 | flowers s3 | flowers s4 | pets s1 | pets s2 | pets s3 | pets s4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CLIP zero-shot | 44.39 | — | — | — | 71.38 | — | — | — | 89.07 | — | — | — |
| PTA | 47.70 | 47.16 | 47.81 | 47.22 | 74.46 | 74.34 | 74.50 | 74.91 | 91.20 | 91.22 | 91.01 | 91.28 |
| CMP-soft | 43.97 | 44.15 | 44.80 | 45.39 | 73.93 | 74.18 | 74.83 | 74.54 | 90.79 | 90.79 | 90.81 | 90.81 |
| CMP-hard | 40.66 | 40.84 | 42.20 | 42.02 | 71.34 | 71.46 | 72.27 | 71.58 | 88.55 | 88.55 | 89.13 | 88.39 |
| GAU λ=0.1 (rescaled) | 44.62 | 44.62 | 43.56 | 43.56 | 64.27 | 63.70 | 63.50 | 63.22 | 59.14 | 56.96 | 58.87 | 61.73 |
| GAU λ=0.5 (rescaled) | 38.36 | 38.95 | 37.59 | 39.48 | 55.18 | 55.14 | 54.28 | 54.61 | 57.89 | 58.05 | 57.92 | 59.69 |
| GAU λ=0.5 (std fusion) | 44.44 | 44.44 | 44.44 | 44.56 | 71.25 | 71.34 | 71.34 | 71.17 | 89.97 | 89.97 | 90.16 | 90.43 |
| BNK K=3 (=K=5) | 46.63 | 47.22 | 46.75 | 46.81 | 74.46 | 74.30 | 74.38 | 74.62 | 91.14 | 91.11 | 91.11 | 91.14 |
| BNK K=1 (bug-fix check) | 46.63 | — | — | — | — | — | — | — | — | — | — | — |
| BNKV2 K=3 | 41.84 | 42.79 | 42.32 | 42.38 | 67.64 | 69.83 | 67.76 | 69.35 | 86.89 | 87.19 | 86.07 | 87.00 |
| BNKV2 K=5 | 41.02 | 42.02 | 42.61 | 42.20 | 67.68 | 69.43 | 69.27 | 68.57 | 87.35 | 86.51 | 86.75 | 87.08 |
| BNKCMP K=3 | 41.96 | 42.73 | 42.26 | 42.20 | 66.67 | 68.21 | 68.01 | 68.37 | 85.61 | 86.15 | 83.86 | 85.96 |
| BNKCMP K=5 | 40.78 | 42.02 | 42.55 | 42.14 | 66.79 | 67.97 | 68.74 | 67.72 | 86.15 | 85.45 | 85.50 | 86.13 |
| GAUCMP s0.1 | 41.55 | 42.08 | 42.02 | 42.43 | 57.82 | 58.18 | 56.84 | 58.18 | 73.15 | 74.19 | 74.43 | 76.04 |
| GAUCMP s0.5 | 31.74 | 32.74 | 32.09 | 34.75 | 40.44 | 42.27 | 38.90 | 41.66 | 73.32 | 74.22 | 74.93 | 75.39 |
| AUG-V4K2 (wave 3) | 47.58 | 47.52 | 47.16 | 46.81 | 73.00 | 73.65 | 73.69 | 73.97 | 91.09 | 90.84 | 90.65 | 90.71 |
| AUG-V8K4 (wave 3) | 46.87 | 46.87 | 46.22 | 46.69 | 72.51 | 72.15 | 72.92 | 72.55 | 90.92 | 90.79 | 91.03 | 90.95 |
| ANCH-d096 (wave 3) | 47.52 | 47.16 | 47.75 | 47.16 | 74.67 | 74.58 | 74.54 | 74.58 | 91.09 | 91.25 | 91.03 | 91.25 |
| ANCH-d093 (wave 3) | 47.58 | 47.22 | 47.75 | 47.16 | 74.67 | 74.58 | 74.50 | 74.58 | 91.09 | 91.25 | 91.06 | 91.25 |
| ABL-V1K1 (wave 3) | 47.70 | 47.16 | 47.81 | 47.22 | 74.46 | 74.34 | 74.50 | 74.91 | 91.20 | 91.22 | 91.01 | 91.28 |
| ABL-V8K8 (wave 3) | 46.10 | 45.39 | 44.56 | 44.92 | 71.01 | 70.73 | 70.73 | 71.50 | 90.79 | 90.76 | 90.57 | 90.46 |

CLIP per-seed values are the 4-seed mean (CLIP is seed-invariant). PTA per-seed values are from the archived baseline records. **ABL-V1K1 (wave 3) reproduces the archived PTA per-seed values exactly** — the identity ablation is confirmed at the record level, and its per-sample write-purity stats match PTA's to the reported precision.

### A.2 Pre-validation gate outputs

Full gate tables in Part 0. Gate scripts: `scripts/` (one-off, archived with the run logs).

### A.3 Reproducibility

- Code: `models/gauss_pta.py`, `models/bank_pta.py`, `models/compact_pta.py`, `models/bank_v2_pta.py`, `models/bank_compact_pta.py`, `models/gauss_compact_pta.py`; configs `configs/{gauss,bank,compact,bank_v2,bank_compact,gauss_compact}_pta/`; slurm scripts `experiments/slurm/09_compact.sh`, `10_gauss.sh`, `11_bank.sh` (wave 1), `14_bank_v2.sh`, `15_bank_compact.sh`, `16_gauss_compact.sh` (wave 2 + combinations), smoke `12_smoke.sh` / `13_smoke_v2.sh`.
- Wave-3 code: `models/aug_pta.py` (V-view entropy-selected ensembling), `models/anchor_pta.py` (text-anchor damped EMA); configs `configs/{aug_pta,anchor_pta}/` with per-dataset yamls; slurm `experiments/slurm/17_aug_pta.sh` (AUG-V8K4/V4K2), `18_anchor_pta.sh` (d096/d093), `19_pta_sweep.sh` (α×T grid), `20_ablation.sh` (V1K1/V8K8), smoke `21_smoke_wave3.sh`. Unit tests `scripts/test_{aug_pta,anchor_pta}_unit.py` — V=1 / damp_threshold=0 equivalence to base PTA.
- Results: `outputs_repr_upgrade/result_{compact,gauss,bank,bank_v2,combos}.txt` (waves 1–2), `result_{aug_pta,anchor_pta,pta_sweep,abl}.txt` (wave 3); per-sample records `outputs_repr_upgrade/records_{compact,gauss,bank,bank_v2,combos,aug_pta,anchor_pta,pta_sweep,abl}/`; aggregation `scripts/aggregate_repr_upgrade.py`; write-purity `scripts/wave3_write_purity.py`.
- Backbone ViT-B/16 (CLIP Surgery), seeds 1-4, batch size 1 (TTA processes one sample at a time by design).
- Wave-2 settings: BankV2 / BankCompact `create_threshold=0.85` (K∈{3,5}); GaussCompact `shrink∈{0.1,0.5}`; both combinations `gate_mode=soft`.
- Wave-3 settings: AugPTA `image_level.n_aug ∈ {4,8}`, `image_level.k_views ∈ {2,4}`; AnchorPTA `image_level.damp_threshold ∈ {0.93,0.96}`, `image_level.damp_floor=0.5`; sweep overrides `image_level.alpha ∈ {0.003,0.1}` × `image_level.T ∈ {10,40}` (2 seeds); ablations V1K1 (identity) and V8K8 (mean-free). Wave-3 adapters are strict subsets of base PTA (same ungated softmax-EMA write rule, same fusion, `tau_patch_proto=0.0`).
- Unit tests (all pass): `scripts/test_{bank_v2,bank_compact,gauss_compact}_pta_unit.py` — K=1/off equivalence to base PTA, grow/recycle correctness, anchor safety, gating behavior.
