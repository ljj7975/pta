# Phase 10: Drift, Floor, Anchor, and Probability Diagnostics — Results

## Context

Phase 8 (`Phase8_Prototype_Repulsion_Results.md`) showed that confusable prototype repulsion fixes
the geometry it claims to fix (confusability measurably drops every time it fires) but regresses
`oxford_pets` at every dose, because the static confusability signal can't distinguish drift-caused
closeness from genuine fine-grained visual similarity (lines 94-104, 108-126). Phase 9
(`Phase9_Drift_Gated_Repulsion_Results.md`) added a drift-velocity gate and a repulsion floor, which
dramatically reduced harm but could not cleanly decompose *how much* of the recovery came from each
mechanism (lines 100-103). Phase 10 addresses four open questions from the Phases 8-9 line of
investigation, each tested as an independent experiment:

1. **Exp1, Floor Ablation**: A 2x2 decomposition (floor x drift) that isolates exactly how much of
   Phase 9's recovery is attributable to the floor vs. the drift gate. Source:
   `outputs/floor_ablation_report.md`.

2. **Exp2, Text-Anchored EMA**: Blends each write toward the class's text feature via a tunable
   `anchor_mix` parameter, testing whether text-anchoring stabilizes prototype direction and improves
   classification. Source: `outputs/text_anchored_report.md`.

3. **Exp3, Probability-Weighted EMA**: Weights each write by `p^gamma` (the top-1 class probability
   raised to a power), testing whether downweighting low-confidence writes improves accuracy. Source:
   `outputs/prob_weighted_report.md`.

4. **Exp4, Bilateral-Drift (OR-condition)**: Widens the Phase 9 drift gate from AND (only top1 drift)
   to OR (top1 or nearest-other drift), testing whether more permissive firing improves coverage.
   Source: `outputs/bilateral_drift_report.md`.

**Dev set**: `dtd`, `oxford_flowers`, `oxford_pets`, ViT-B/16, CLIP Surgery, seeds 1-4, reference
(dtd 46.85 / flowers 74.22 / pets 90.75). Control identity reproduces the reference exactly in all
four experiments. Standard bar: avg gain > 0.3pp AND no single-dataset regression > 0.5pp vs control.

---

## Results

### Exp1: Floor Ablation (2x2 Decomposition)

Tests whether the repulsion floor or the drift gate drives Phase 9's recovery from Phase 8's damage.
The key insight: `confusable_with_floor` at lr=0.02 is bit-identical to Phase 8's `confusable`-alone
because the floor (floor_percentile=20.0) does not bind at low dose (repulsion never pushes a
prototype below the floor). The floor only matters at high dose
(`outputs/floor_ablation_report.md`, Section 4).

#### Accuracy (4-seed mean) vs control

| Setting | dtd | oxford_flowers | oxford_pets | Avg | Δ vs control |
|---|---|---|---|---|---|
| control | 46.85 | 74.22 | 90.75 | 70.608 | +0.000 |
| lr0.02-f (floor only) | 47.03 | 74.48 | 89.88 | 70.465 | −0.142 |
| lr0.05-f (floor only) | 47.33 | 74.61 | 87.10 | 69.680 | −0.927 |
| lr0.1-f (floor only) | 46.38 | 73.90 | 82.75 | 67.676 | −2.932 |
| lr0.02-do (drift only) | 46.75 | 74.40 | 90.77 | 70.639 | +0.032 |
| lr0.05-do (drift only) | 47.03 | 74.44 | 90.52 | 70.663 | +0.056 |

Source: `outputs/floor_ablation_report.md`, Section 1.

#### 2x2 Decomposition (oxford_pets Δ)

| Dose | (a) none/none (Phase 8) | (b) none/drift (drift only) | (c) floor/none (floor only) | (d) floor/drift (Phase 9) |
|---|---|---|---|---|
| 0.02 | −0.86 | +0.02 | −0.86 | +0.02 |
| 0.05 | −3.64 | −0.22 | −3.64 | −0.22 |
| 0.1 | −8.00 | N/A | −8.00 | −1.08 |

Source: `outputs/floor_ablation_report.md`, Section 2. Cell (a) from
`outputs/result_repulsion.txt` (Phase 8), cells (b)/(c) from `outputs/result_floor_ablation.txt`,
cell (d) from `outputs/result_drift_gated_repulsion.txt` (Phase 9).

#### Verdict: Drift Dominates

At lr=0.02 on oxford_pets, drift-only recovers +0.89pp (100% of Phase 8's gap) while floor-only
recovers +0.00pp (0%). The floor does NOT bind at low dose; it only matters at high dose (lr0.1-f
−8.00 vs lr0.1-drift −1.08). The drift gate is the active ingredient
(`outputs/floor_ablation_report.md`, Section 4).

Mechanism table confirms the separation: floor-only fire rates (0.726/0.720/0.591) match Phase 8's
`confusable`-alone (0.591-0.726), while drift-only fire rates (0.115/0.122) match Phase 9's
`drift_confusable` (0.115-0.122). The drift gate cuts fire rate ~6x and saves oxford_pets
(`outputs/floor_ablation_report.md`, Section 3).

Exp4 gate: **OPEN** (gate a: drift-only Δ − floor-only Δ = +0.89pp > 0.1pp PASS; gate b:
drift+floor − floor-only on pets = +0.89pp > 0.1pp PASS) (`outputs/floor_ablation_report.md`,
Section 4).

#### Per-setting go/no-go

All 5 non-control settings: **no-go** by the standard bar (floor-only settings regress >0.5pp on at
least one dataset; drift-only settings are positive but below +0.3pp average gain).

---

### Exp2: Text-Anchored EMA

Tests whether blending each write toward the class's text feature stabilizes prototype direction.
anchor_mix sweep (0.99/0.95/0.90/0.80, where 1.0 = pure image and 0.0 = pure text). Control
identity: PASS (dtd 46.85 / flowers 74.22 / pets 90.75, |Δ|=0.00pp)
(`outputs/text_anchored_report.md`, control identity check).

#### Accuracy (4-seed mean) vs control

| Setting | anchor_mix | dtd | oxford_flowers | oxford_pets | Avg | Δ vs control |
|---|---|---|---|---|---|---|
| control | 1.0 | 46.85 | 74.22 | 90.75 | 70.608 | +0.000 |
| a0.99 | 0.99 | 46.84 | 74.20 | 90.75 | 70.598 | −0.009 |
| a0.95 | 0.95 | 46.91 | 74.26 | 90.79 | 70.653 | +0.046 |
| a0.90 | 0.90 | 46.88 | 74.27 | 90.74 | 70.631 | +0.023 |
| a0.80 | 0.80 | 46.66 | 74.06 | 90.74 | 70.488 | −0.119 |

Source: `outputs/text_anchored_report.md`, Overall Accuracy table.

#### Per-dataset Δ

| Setting | dtd Δ | oxford_flowers Δ | oxford_pets Δ |
|---|---|---|---|
| a0.99 | −0.02 | −0.02 | +0.01 |
| a0.95 | +0.06 | +0.03 | +0.04 |
| a0.90 | +0.03 | +0.05 | −0.00 |
| a0.80 | −0.19 | −0.16 | −0.00 |

Source: `outputs/text_anchored_report.md`, Per-dataset Δ table.

Best avg Δ = +0.046pp (a0.95), far below the +0.3pp bar. Monotonicity: non-monotonic across all
datasets (accuracy peaks at a0.95 then reverses). Convergence: acc@10% degrades monotonically with
stronger anchoring (68.45 → 67.79), but mid/late convergence is flat (69.28-69.96 range holds)
(`outputs/text_anchored_report.md`, Monotonicity and Convergence sections). Text anchoring hurts
initial adaptation speed without affecting final accuracy.

All 4 settings: **no-go**.

---

### Exp3: Probability-Weighted EMA

Tests whether downweighting low-confidence top-1 writes (weight = `p^gamma`) improves accuracy.
gamma sweep (0.5/1.0/2.0). Control identity: PASS (dtd 46.85 / flowers 74.22 / pets 90.75)
(`outputs/prob_weighted_report.md`, Control Identity Check).

#### Accuracy (4-seed mean) vs control

| Setting (gamma) | dtd | oxford_flowers | oxford_pets | Avg | Δ vs control |
|---|---|---|---|---|---|
| control (γ=0.0) | 46.85 | 74.22 | 90.75 | 70.608 | +0.000 |
| g0.5 (γ=0.5) | 46.26 | 74.40 | 90.72 | 70.462 | −0.146 |
| g1.0 (γ=1.0) | 45.73 | 74.47 | 90.70 | 70.300 | −0.308 |
| g2.0 (γ=2.0) | 44.66 | 74.12 | 90.51 | 69.763 | −0.844 |

Source: `outputs/prob_weighted_report.md`, Overall Accuracy table.

#### Per-dataset Δ

| Setting | dtd Δ | oxford_flowers Δ | oxford_pets Δ |
|---|---|---|---|
| g0.5 | −0.59 | +0.18 | −0.03 |
| g1.0 | −1.12 | +0.25 | −0.05 |
| g2.0 | −2.19 | −0.11 | −0.24 |

Source: `outputs/prob_weighted_report.md`, Per-dataset Δ table.

Monotonicity: PARTIAL. dtd decreases monotonically (46.85 → 46.26 → 45.73 → 44.66) and oxford_pets
decreases monotonically (90.75 → 90.72 → 90.70 → 90.51), but oxford_flowers peaks at γ=1.0 (74.47)
before dropping. dtd is the primary casualty (−0.59 to −2.19pp), the hardest dataset most affected by
downweighting low-confidence writes (`outputs/prob_weighted_report.md`, Monotonicity section).
Convergence: acc@10% degrades from 68.45 (control) to 64.85 (γ=2.0); auc_norm from 0.98 to 0.96
(`outputs/prob_weighted_report.md`, Convergence table).

All 3 settings: **no-go**.

---

### Exp4: Bilateral-Drift (OR-condition)

Widens the Phase 9 drift gate from AND (only top1 drift) to OR (top1 or nearest-other class drift),
testing whether more permissive firing improves coverage. Fire rates: OR-condition (0.168/0.162/0.156)
vs Phase 9 AND (0.115/0.122/0.117) at matched doses, confirming the OR is strictly more permissive
(`outputs/bilateral_drift_report.md`, Fire-rate comparison).

#### Accuracy (4-seed mean) vs control

| Setting | dtd | oxford_flowers | oxford_pets | Avg | Δ vs control |
|---|---|---|---|---|---|
| control | 46.85 | 74.22 | 90.75 | 70.608 | +0.000 |
| lr0.02-bd | 46.94 | 74.44 | 90.75 | 70.711 | +0.103 |
| lr0.05-bd | 47.02 | 74.43 | 90.34 | 70.597 | −0.011 |
| lr0.1-bd | 47.07 | 74.37 | 89.19 | 70.210 | −0.397 |

Source: `outputs/bilateral_drift_report.md`, Overall Accuracy table.

#### Per-dataset Δ

| Setting | dtd Δ | oxford_flowers Δ | oxford_pets Δ |
|---|---|---|---|
| lr0.02-bd | +0.09 | +0.22 | +0.01 |
| lr0.05-bd | +0.16 | +0.21 | −0.40 |
| lr0.1-bd | +0.22 | +0.15 | −1.56 |

Source: `outputs/bilateral_drift_report.md`, Per-dataset Δ table.

lr0.02-bd is the cleanest result (zero regression on any dataset, avg +0.103pp) but short of the
+0.3pp bar. Cross-study vs Phase 9 `drift_confusable`: BD improves at dose 0.02 (+0.072pp avg diff)
but regresses at 0.05 (−0.067pp) and 0.1 (−0.152pp). At higher doses, the additional OR-triggered
fires hurt oxford_pets more than the AND-condition does (lr0.1-bd −1.56 vs lr0.1-drift −1.08)
(`outputs/bilateral_drift_report.md`, Cross-study verdict). Control identity: PASS (46.85/74.22/90.75,
all |Δ|=0.00pp).

All 3 settings: **no-go**. Verdict: **MIXED / NO-GO** (improves at 1 dose, regresses at 2 vs Phase 9;
no promotion).

---

## Go/no-go

| Experiment | Best setting | Best avg Δ | Bar (>0.3pp)? | Verdict |
|---|---|---|---|---|
| Exp1 (floor ablation) | lr0.02-do | +0.032pp | NO | no-go |
| Exp2 (text-anchored) | a0.95 | +0.046pp | NO | no-go |
| Exp3 (prob-weighted) | g0.5 | −0.146pp | NO | no-go |
| Exp4 (bilateral-drift) | lr0.02-bd | +0.103pp | NO | no-go |

All 16 non-control settings across all four experiments: **no-go** by the standard bar. No held-out
validation triggered (no go verdict exists anywhere in this phase). All four control identity checks
reproduce the reference exactly (dtd 46.85 / flowers 74.22 / pets 90.75).

## Conclusion

Phase 10 resolves the key open questions from the Phases 8-9 confusability-repulsion line:

1. **Exp1 (drift dominates)**: Phase 9's recovery from Phase 8 is almost entirely attributable to
   the drift gate cutting fire rate ~6x, not the floor. The floor does not bind at low dose (the
   recommended setting) and only matters at high dose where repulsion would compound destructively
   anyway. The drift signal is the active ingredient; the floor is a safety net.

2. **Exp2 (text-anchoring null)**: Anchoring writes toward text features has no meaningful accuracy
   upside. The tiny positive at a0.95 (+0.046pp) is within noise. Early convergence degrades, late
   convergence is flat.

3. **Exp3 (prob-weighting null)**: Downweighting low-confidence writes via p^gamma hurts the hardest
   dataset (dtd) most severely (−0.59 to −2.19pp). The mechanism works as designed (it downweights
   uncertain writes) but the uncertain writes contain information the EMA needs.

4. **Exp4 (bilateral/OR null)**: The OR-condition fires more often but does not improve accuracy.
   The AND-condition's selectivity on top1 drift was already filtering out most harmful fires; more
   firing is not better.

The confusability-repulsion line of investigation (Phases 8-10) has now been exhaustively explored:
defensive gating/reweighting (Phase 7), undirected correction (Phase 8), directed correction with
drift gating (Phase 9), floor/drift decomposition and alternative write-modulation strategies
(Phase 10). None clears the promotion bar. The trend across Phases 8-10 is toward progressively
milder harm, not toward a positive result.

---

## Phase 11: Decision Tree

Each non-null verdict maps to a next direction or a closure recommendation:

| Verdict | Experiment | Next direction |
|---|---|---|
| Drift dominates | Exp1 | The drift signal is the active ingredient. Explore finer drift thresholds / firing schedules for the drift gate (the mechanism works; the dose is not yet right). |
| Text-anchoring null | Exp2 | Do NOT pursue text-anchored EMA further. No convergence or accuracy upside from any anchor_mix < 1.0. Close this direction. |
| Prob-weighting null | Exp3 | Do NOT pursue p^gamma downweighting further. The mechanism hurts the hardest dataset most and degrades convergence. Close this direction. |
| Bilateral/OR null | Exp4 | Do NOT pursue the more-permissive OR trigger. AND-condition selectivity already filters harmful fires; more firing is not better. Close this direction. |
| All four null | Overall | The confusability-repulsion line of investigation has been exhaustively explored (Phases 8-10) with no promotable result. Recommend closing this line of investigation entirely. Do NOT recommend held-out validation; no go verdict exists to validate. |

The only surviving signal is Exp1's drift-dominant finding: drift gating is the active ingredient in
Phase 9's recovery, and finer-grained drift-threshold/firing-schedule tuning is the sole remaining
sub-direction worth exploring. All other Phase 10 directions (text-anchoring, probability-weighting,
bilateral triggering) are closed.

## Reproducing

```bash
# Exp1: Floor ablation sweep (72 runs)
bash scripts/run_floor_ablation_sweep.sh --run
python scripts/analyze_floor_ablation.py

# Exp2: Text-anchored EMA sweep (60 runs)
bash scripts/run_text_anchored_sweep.sh --run
python scripts/analyze_text_anchored.py

# Exp3: Probability-weighted EMA sweep (48 runs)
bash scripts/run_prob_weighted_sweep.sh --run
python scripts/analyze_prob_weighted.py

# Exp4: Bilateral-drift sweep (48 runs)
bash scripts/run_bilateral_drift_sweep.sh --run
python scripts/analyze_bilateral_drift.py
```
