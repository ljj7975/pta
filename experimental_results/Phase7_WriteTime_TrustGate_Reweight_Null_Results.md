# Phase 7: Write-Time Trust Gate / Reweight, Driven by View-Consistency and Prototype-Confusability

## Context

`PatchModPTA_Purity_Separability_Trust_Analysis.md` (Parts 4a/4b) established that write-time trust
levers — a hard gate (`models/write_gate_pta.py`) and a soft two-sided reweight
(`models/reweight_pta.py`) built on the **patch-vote** signal — both fail to beat plain top-1-write PTA:
gating collapses fine-grained accuracy (flowers 74.5 → 42%) and reweighting never beats the control on
final accuracy or convergence speed. The diagnosed cause was specific to that signal: the topk20 patch
vote's own standalone accuracy is *below* CLIP's, so "disagreement" throws away good writes more often
than bad ones.

`Phase4-6_Calibration_Confusability_ViewConsistency_Null_Results.md` then validated two *new* diagnostics
that don't share that flaw — each tests CLIP's own robustness or the prototype bank's own geometry,
not an alternative classifier:

- **Multi-view consistency**: agreement between CLIP's prediction on the original image and `n_views-1`
  augmented copies. Failed only as a **read-time** reweighting lever (mechanism-specific: near-total
  overlap between "untrusted" and "tie" samples).
- **Prototype confusability**: whether a class's prototype sits too close to another class's. Failed only
  as a **permanent per-class freeze** (monotonic regression, diagnosed cause was the permanence itself,
  not the signal).

Neither had been tried as a **per-sample, freshly re-evaluated write-time gate/reweight** — the exact
mechanism family that failed for patch-vote. This phase runs that test, for each signal **independently**
and **combined** (AND'd together), through both of Part 4a/4b's mechanisms.

**Dev set**: `dtd`, `oxford_flowers`, `oxford_pets`, ViT-B/16, CLIP Surgery, seeds 1-4, single-class
top-1 EMA write (identical to Part 4a/4b's mechanism, never base PTA's multi-class write).

---

## Signals (computed fresh every sample, before the write)

**View** (`_compute_view_clean`, reused from `models/view_consistency_pta.py`'s augmentation):
```python
top1 = int(clip_logits.argmax(dim=-1).item())
aug_views = [_augment_image(images) for _ in range(n_views - 1)]      # n_views = 4
aug_feats = _safe_normalize(encoder.encode_image(torch.cat(aug_views, dim=0).cuda()).float())
aug_preds = (100.0 * aug_feats @ text_embeddings.float()).argmax(dim=-1)
view_clean = float((aug_preds == top1).float().mean()) >= agreement_thresh   # agreement_thresh = 1.0
```

**Confusability** (`_compute_proto_clean`) — re-evaluated fresh every sample, **not** a permanent freeze:
```python
proto_norm = _safe_normalize(prototype_state.float(), dim=-1)
sims = proto_norm @ proto_norm.t(); sims.fill_diagonal_(-1.0)
nearest_other, _ = sims.max(dim=1)                                    # [C]
written = prototype_state.float().norm(dim=-1) > 1e-6
if written.sum() >= 5 and written[top1]:
    thresh = torch.quantile(nearest_other[written], 66 / 100.0)       # Phase 2a's validated tercile boundary
    proto_clean = not bool(nearest_other[top1] > thresh)
else:
    proto_clean = True     # not enough written classes yet, or class never written
```
Unlike `models/confusability_gated_pta.py`, `proto_clean` has no memory: if the bank's geometry later
shifts so a class is no longer confusable, its next write proceeds normally. This isolates whether the
earlier freeze study's failure was caused by permanence specifically.

**Combined** (`signal_source=both`): `clean = view_clean and proto_clean`.

Both new adapters skip all signal computation when the outcome can't depend on it — `baseline` (Study A)
and `boost==boost_down` (Study B control) — so the sweep never pays for the multi-view forward passes or
the confusability matmul when they're not needed.

---

## Study A — Write-Gate (`models/trust_write_gate_pta.py`)

Mirrors Part 4a exactly, replacing the patch-vote signal with `signal_source ∈ {view, confusability,
both}`:

| write_gate | condition |
|---|---|
| `baseline` | always write `argmax(clip)` (control, signal-independent) |
| `clean` | write iff the selected signal's clean condition holds |
| `confident_and_clean` | `confident` (`softmax(clip).top1 − .top2 ≥ 0.2`) AND `clean` |

7 settings (`baseline` + 2 gate modes × 3 signal sources) × 3 datasets × 4 seeds = 84 runs, all completed
with 0 failures.

### Accuracy (4-seed mean)

| Setting | dtd | oxford_flowers | oxford_pets | Avg | Δ vs baseline |
|---|---|---|---|---|---|
| baseline | 46.85 | 74.22 | 90.75 | 70.608 | +0.000 |
| clean-conf | 46.34 | 73.02 | 90.15 | 69.834 | −0.773 |
| cac-conf | 41.52 | 73.92 | 90.07 | 68.503 | −2.104 |
| clean-view | 38.18 | 60.74 | 87.75 | 62.223 | −8.385 |
| cac-view | 35.42 | 60.10 | 87.49 | 61.003 | −9.604 |
| clean-both | 38.14 | 60.73 | 87.61 | 62.157 | −8.451 |
| cac-both | 35.14 | 60.01 | 87.19 | 60.778 | −9.830 |

All 6 non-baseline settings: **no-go** (bar: >0.3pp avg gain, no >0.5pp regression on any dataset).

### Write rate (fraction of samples where the gate fires)

Ranged from 0.616–0.663 (`clean-conf`, mildest) down to 0.101–0.181 (`cac-both`, most restrictive), with
the expected ordering `both ≤ view ≤ confusability` confirming the AND-combination logic is correct.

### Mechanism

**Confusability gating is far milder than view-based gating** (−0.8 to −2.1pp vs. −8.4 to −9.8pp), and
milder than the earlier *permanent* freeze from Phase 2 (−2.2 to −10.0pp) — removing permanence
substantially reduces the damage, but it does not flip the sign positive: even a freshly re-evaluated
confusability gate still blocks enough good writes to net negative. View-based gating is the most
destructive lever tested in this campaign at the write side, consistent with `agreement_thresh=1.0`
being an extremely strict bar (any single augmented view disagreeing blocks the write) applied to signal
that Phase 4-6 showed correlates with correctness but at a much coarser resolution than per-sample
gating can tolerate.

---

## Study B — Two-Sided Write Reweight (`models/trust_reweight_pta.py`)

Mirrors Part 4b.5: writes are never dropped, only the EMA weight is scaled —
`applied_factor = boost if (confident and clean) else boost_down`. 13 settings (`(1.0,1.0)` control +
4 `(boost, boost_down)` points × 3 signal sources) × 3 datasets × 4 seeds = 156 runs, all completed with
0 failures.

### Accuracy (4-seed mean) vs. `control` (1.0, 1.0)

| Setting | dtd | oxford_flowers | oxford_pets | Avg | Δ vs control |
|---|---|---|---|---|---|
| control | 46.85 | 74.22 | 90.75 | 70.608 | +0.000 |
| b1.0d0.5-conf | 46.70 | 74.45 | 90.68 | 70.613 | +0.006 |
| b1.0d0.1-conf | 45.30 | 74.23 | 90.21 | 69.912 | −0.695 |
| b2.0d0.5-conf | 46.38 | 74.48 | 90.46 | 70.442 | −0.165 |
| b2.0d0.1-conf | 45.24 | 74.07 | 90.06 | 69.792 | −0.816 |
| b1.0d0.5-view | 46.50 | 74.36 | 90.68 | 70.513 | −0.095 |
| b1.0d0.1-view | 44.21 | 72.81 | 89.66 | 68.893 | −1.714 |
| b2.0d0.5-view | 45.70 | 74.34 | 90.56 | 70.198 | −0.409 |
| b2.0d0.1-view | 43.45 | 72.49 | 89.09 | 68.345 | −2.263 |
| b1.0d0.5-both | 46.79 | 74.20 | 90.63 | 70.541 | −0.067 |
| b1.0d0.1-both | 44.43 | 72.69 | 89.46 | 68.858 | −1.749 |
| b2.0d0.5-both | 46.07 | 74.20 | 90.39 | 70.220 | −0.388 |
| b2.0d0.1-both | 43.97 | 72.33 | 88.96 | 68.420 | −2.188 |

`b1.0d0.5-conf` lands at +0.006pp — indistinguishable from noise, not a gain. All 12 settings: **no-go**
on the same bar as Study A.

### Convergence (pooled mean across 3 datasets × 4 seeds)

| Setting | acc@10% | acc@25% | acc@50% | acc@75% | final | auc_norm |
|---|---|---|---|---|---|---|
| control | 68.45 | 69.28 | 69.59 | 69.96 | 70.61 | 0.9787 |
| b1.0d0.5-conf | 68.01 | 69.13 | 69.55 | 69.96 | 70.61 | 0.9773 |
| b1.0d0.1-conf | 67.03 | 68.25 | 68.69 | 69.27 | 69.91 | 0.9731 |
| b2.0d0.5-conf | 67.98 | 69.00 | 69.34 | 69.85 | 70.44 | 0.9775 |
| b2.0d0.1-conf | 66.41 | 67.89 | 68.51 | 69.11 | 69.79 | 0.9706 |
| b1.0d0.5-view | 67.17 | 68.66 | 69.32 | 69.77 | 70.51 | 0.9731 |
| b1.0d0.1-view | 59.84 | 64.82 | 66.70 | 67.85 | 68.89 | 0.9403 |
| b2.0d0.5-view | 67.12 | 68.33 | 68.95 | 69.45 | 70.20 | 0.9734 |
| b2.0d0.1-view | 59.19 | 63.96 | 65.95 | 67.28 | 68.35 | 0.9383 |
| b1.0d0.5-both | 67.10 | 68.74 | 69.34 | 69.80 | 70.54 | 0.9732 |
| b1.0d0.1-both | 59.64 | 64.72 | 66.78 | 67.86 | 68.86 | 0.9396 |
| b2.0d0.5-both | 66.91 | 68.33 | 68.97 | 69.48 | 70.22 | 0.9727 |
| b2.0d0.1-both | 58.86 | 63.76 | 66.06 | 67.31 | 68.42 | 0.9356 |

No setting improves `acc@10%` or `auc_norm` over control — every point is flat-to-worse on convergence
speed as well as final accuracy. There is no partial/mechanistic win to report here: unlike the earlier
read-time reweighting study, down-weighting untrusted writes doesn't even buy faster early adaptation in
exchange for a final-accuracy cost — it costs both, monotonically, as `boost_down` decreases.

### Matched-point comparison — view vs. confusability vs. both (avg accuracy across datasets)

| Point | confusability | view | both |
|---|---|---|---|
| (1.0, 0.5) | 70.613 | 70.513 | 70.541 |
| (1.0, 0.1) | 69.912 | 68.893 | 68.858 |
| (2.0, 0.5) | 70.442 | 70.198 | 70.220 |
| (2.0, 0.1) | 69.792 | 68.345 | 68.420 |

At every matched point, confusability is the mildest signal, view and both are close to each other (both
never separates meaningfully from view alone, since confusability rarely disagrees with view once view
has already filtered), and severity scales directly with `boost_down`'s distance from 1.0 — the reweight
never reverses this ordering into a gain regardless of how gently the untrusted writes are down-weighted.

---

## Cross-Cutting Summary: Three Signals Through the Same Write-Time Levers

| Signal | Write-gate (`clean`) | Write-reweight (mildest point) | Standalone diagnostic strength |
|---|---|---|---|
| Patch-vote (Part 4a/4b) | −4.5pp avg (flowers −32pp) | never beats control | 30+pp purity gap, but *own* accuracy below CLIP's |
| Prototype confusability (this phase) | −0.8pp avg | +0.006pp (noise) | +9 to +16.5pp tercile gap (Phase 4-6) |
| Multi-view consistency (this phase) | −8.4pp avg | −0.095pp | +16.6 to +27.4pp purity gap (Phase 4-6), strongest of the three |
| Both combined (this phase) | −8.5pp avg (≈ view alone) | −0.067pp (≈ view alone) | — |

Three observations generalize across all four signal/mechanism combinations tried across this and the
prior phase:

1. **Diagnostic strength does not predict write-time usefulness, and the relationship is inverted.** The
   two new signals are markedly *stronger* standalone diagnostics than patch-vote, yet all three still
   fail as write-time levers. Multi-view — the single strongest purity gap in the whole campaign — produces
   the single most destructive write-gate result of the three signals. A signal that cleanly separates
   correct from incorrect predictions on average is not the same as a signal that identifies *which
   specific writes* are safe to suppress or down-weight; every write blocked or discounted removes
   information the EMA needs, and these signals evidently flag many writes that are still net-useful to
   the running prototype even when the frame-level prediction is itself imperfect.

2. **Combining signals never helps and never meaningfully separates from the stronger individual signal.**
   `both` tracks `view` almost exactly in every table above (Study A: −8.45 vs −8.39pp; Study B: −0.067 vs
   −0.095pp), because the AND-combination is dominated by whichever signal is more restrictive — it cannot
   rescue a failing signal by pairing it with a milder one, and confusability being milder doesn't dilute
   view's damage since AND-ing only adds restriction, never removes it.

3. **Removing permanence, and removing hardness, both help — but neither flips the sign.** Confusability's
   permanent freeze regressed −2.2 to −10.0pp (Phase 4-6); the same signal, freshly re-evaluated and used
   only in a per-sample gate, regresses just −0.8 to −2.1pp; used as a soft reweight instead of a hard
   gate, it's statistically flat (as mild as +0.006pp, as costly as −0.8pp). Each relaxation (permanent →
   per-sample, hard gate → soft reweight) reliably narrows the loss, but three different signals, four
   different mechanisms (permanent freeze, per-sample hard gate, per-sample two-sided reweight, read-time
   reweight), and two full phases of this campaign have now converged on the same conclusion: **write-time
   trust levers built on the signals investigated so far do not have room to help PTA's top-1-write
   mechanism** — they only vary in how much they hurt it. The prototype bank's accuracy is evidently
   already close to what the write policy can extract from `argmax(clip)`, and filtering or discounting
   writes trades away more good adaptation than it prevents bad adaptation, regardless of which of these
   three signals does the filtering.

---

## Reproducing

```bash
bash scripts/run_trust_write_gate_sweep.sh --run   # Study A: 84 runs, ~15-20 min
python scripts/analyze_trust_write_gate.py

bash scripts/run_trust_reweight_sweep.sh --run     # Study B: 156 runs, ~55-65 min
python scripts/analyze_trust_reweight.py
```

No setting in either study cleared the go/no-go bar, so held-out validation (`caltech101`, `eurosat`,
`ucf101`; configs already present under `configs/trust_write_gate_pta/` and `configs/trust_reweight_pta/`)
was not run.
