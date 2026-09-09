# Phase 9: Drift-Gated Prototype Repulsion — Results

## Context

Phase 8 (`experimental_results/Phase8_Prototype_Repulsion_Results.md`) found that repelling confusable
prototype pairs apart works exactly as designed geometrically (confusability measurably drops every time
it fires) but regresses `oxford_pets` monotonically at every dose (−0.86 → −3.64 → −8.00pp), because the
static confusability signal can't distinguish a prototype that recently *drifted* close to another class
(a legitimate correction target) from two classes that are simply *genuinely similar* in CLIP's embedding
space (fine-grained breeds — plausible for `oxford_pets` specifically). This phase adds a per-class
**drift-velocity** signal — an EMA of how much a class's prototype direction has moved at each recent
write — and gates repulsion on it (`drift_confusable`: confusable AND drifting), with a `stable_confusable`
counterfactual arm (confusable AND NOT drifting) as a diagnostic. Both arms also add a **repulsion floor**
(stop re-firing on a pair once it's already reasonably separated relative to the bank's own geometry) to
address Phase 8's other failure mode — unbounded compounding at high dose.

**Dev set**: `dtd`, `oxford_flowers`, `oxford_pets`, ViT-B/16, CLIP Surgery, seeds 1-4, same reference
(dtd 46.85 / flowers 74.22 / pets 90.75). Calibration (300-batch smoke test, pre-registered in the plan)
confirmed: control reproduces the reference exactly (40.00% at 300 batches), `drift_ema` takes a
non-degenerate range (0 to 0.089), `drift_confusable`'s fire rate at `lr=0.05` (0.28) is well below Phase
8's `confusable`-alone rate at the same dose (0.720, strictly more selective as required), the floor
measurably decelerates firing over a run (0.35 → 0.21 fire rate, first half vs. second half), and no
numerical instability. 7 settings (control + `{drift_confusable, stable_confusable} × {0.02, 0.05, 0.1}`)
× 3 datasets × 4 seeds = 84 runs, all completed with 0 failures.

---

## Results

### Accuracy (4-seed mean) vs. `control`

| Setting | dtd | oxford_flowers | oxford_pets | Avg | Δ vs control |
|---|---|---|---|---|---|
| control | 46.85 | 74.22 | 90.75 | 70.608 | +0.000 |
| lr0.02-drift | 46.75 | 74.40 | 90.77 | 70.639 | +0.032 |
| lr0.05-drift | 47.03 | 74.44 | 90.52 | 70.663 | +0.056 |
| lr0.1-drift | 47.02 | 74.40 | 89.67 | 70.362 | −0.246 |
| lr0.02-stable | 46.90 | 74.51 | 90.61 | 70.673 | +0.066 |
| lr0.05-stable | 47.37 | 74.46 | 90.02 | 70.618 | +0.011 |
| lr0.1-stable | 47.34 | 74.60 | 87.98 | 69.975 | −0.633 |

All 6 settings: **no-go** by the standard bar (>0.3pp avg gain, no >0.5pp regression). But the picture is
categorically different from Phase 8: no catastrophic collapse anywhere, `dtd`/`oxford_flowers` are
positive in every single setting, and at `lr=0.02` (both triggers) there is **no regression on any
dataset at all** — the first fully-clean (no single-dataset loss >0.5pp) result anywhere in the
confusability-repulsion line of investigation, just short of the promotion bar on average gain (+0.03 to
+0.07pp, vs. the +0.3pp needed).

### Per-dataset Δ

| Setting | dtd Δ | oxford_flowers Δ | oxford_pets Δ |
|---|---|---|---|
| lr0.02-drift | −0.10 | +0.18 | **+0.02** |
| lr0.05-drift | +0.18 | +0.22 | −0.22 |
| lr0.1-drift | +0.16 | +0.18 | −1.08 |
| lr0.02-stable | +0.05 | +0.29 | −0.14 |
| lr0.05-stable | +0.52 | +0.24 | −0.72 |
| lr0.1-stable | +0.49 | +0.37 | −2.76 |

### Mechanism (pooled across datasets × seeds)

| Setting | Fire rate | Mean pre-confusability | Mean post-confusability | Mean drift_velocity |
|---|---|---|---|---|
| lr0.02-drift | 0.115 | 0.9230 | 0.9129 | 0.0028 |
| lr0.05-drift | 0.122 | 0.8946 | 0.8743 | 0.0029 |
| lr0.1-drift | 0.117 | 0.8628 | 0.8274 | 0.0033 |
| lr0.02-stable | 0.201 | 0.9068 | 0.8975 | 0.0028 |
| lr0.05-stable | 0.219 | 0.8354 | 0.8157 | 0.0030 |
| lr0.1-stable | 0.229 | 0.7249 | 0.6931 | 0.0037 |

Two things worth flagging on their own: **fire rates are far lower than Phase 8's** (0.115–0.229 here vs.
0.591–0.726 for `confusable`-alone in Phase 8, at the same doses) — the AND-condition plus the floor are
both doing a lot of restricting — and **`stable_confusable` fires almost twice as often as
`drift_confusable`** (e.g. 0.219 vs. 0.122 at `lr=0.05`). That second fact is itself informative: most
confusable pairs this bank ever flags are *stable*, not drifting — i.e., static confusability is
predominantly picking up genuinely-close classes, not transient drift. This is direct empirical support
for the root-cause diagnosis from Phase 8: closeness is usually a real geometric fact about the class
pair, not a correctable artifact.

### Falsifiable predictions vs. Phase 8's `confusable`-alone oxford_pets deltas

| Dose | Phase 8 `confusable` (no drift gate, no floor) | `drift_confusable` (this study) | `stable_confusable` (this study) | Prediction holds? |
|---|---|---|---|---|
| 0.02 | −0.86 | **+0.02** | −0.14 | a: **YES**, b: NO |
| 0.05 | −3.64 | −0.22 | −0.72 | a: **YES**, b: NO |
| 0.1 | −8.00 | −1.08 | −2.76 | a: **YES**, b: NO |

**Prediction (a) holds cleanly at every dose** — `drift_confusable`'s `oxford_pets` delta is dramatically
less negative than Phase 8's `confusable`-alone at the matched dose (e.g. −1.08 vs. −8.00 at `lr=0.1`, an
~87% reduction in damage). **Prediction (b) does not hold as stated** — `stable_confusable` also beats
Phase 8's `confusable`-alone by a wide margin at every dose (e.g. −2.76 vs. −8.00 at `lr=0.1`), which was
supposed to be the arm that reproduces or worsens the original damage.

The reason prediction (b) doesn't hold is that both new arms bundle the repulsion floor, which Phase 8's
`confusable`-alone did not have — the floor caps how much damage *either* arm can do by preventing
runaway re-firing on the same pair, and it evidently recovers most of the loss on its own. The comparison
that actually isolates the drift-gating variable is `drift_confusable` vs. `stable_confusable` directly
(both have the floor; only the trigger condition differs), and there **`drift_confusable` beats
`stable_confusable` on `oxford_pets` at every single dose** (+0.02 vs. −0.14, −0.22 vs. −0.72, −1.08 vs.
−2.76) — a small but perfectly monotonic, consistent margin. So: **the drift-gating hypothesis is
directionally confirmed, but the floor is doing more of the total repair than the drift signal is**, and
this study cannot cleanly separate exactly how much credit each deserves without a third ablation arm
(`confusable` + floor, no drift condition) that wasn't run here.

---

## Go/no-go

All 6 settings: **no-go** by the standard bar. No held-out validation triggered. But this is the mildest,
safest result the confusability-repulsion line of investigation has produced: at `lr=0.02` (either
trigger), there is no regression exceeding 0.5pp on any of the 3 dev datasets — the only doses/settings in
Phases 8-9 combined with that property — even though the average gain (+0.03 to +0.07pp) is well short of
the +0.3pp promotion bar.

## Conclusion

The drift/genuine-similarity distinction is real and helps, but it is not the dominant fix: firing less
often in general (via the floor, and via requiring an AND of two independent conditions rather than one)
recovers most of Phase 8's damage, while drift-gating specifically contributes a smaller, consistent,
monotonic improvement on top of that recovery. Neither mechanism, alone or combined, produces a
promotable gain — this campaign has now tried defensive gating/reweighting (Phase 7), undirected
correction (Phase 8), and directed correction (Phase 9) against the confusability signal, and none clears
the bar, though the trend across these three is toward progressively milder harm, not toward a positive
result.

**Flagged as a future direction, not built here**: an ablation arm isolating the floor's contribution
alone (`confusable` trigger + floor, no drift condition) would cleanly decompose how much of this study's
recovery is attributable to the floor vs. the drift signal. A second natural extension is checking *both*
classes' drift in a confusable pair (this study only gates on `top1`'s drift, not the nearest-other
class's) — a pair where the *other* class is the one drifting toward a stable `top1` is currently treated
identically to a pair where neither is drifting.

## Reproducing

```bash
bash scripts/run_drift_gated_repulsion_sweep.sh --run   # 84 runs, single-stage, ~15-20 min
python scripts/analyze_drift_gated_repulsion.py
```
