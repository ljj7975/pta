# Hypothesis Validation Findings — Patch-Prototype Growth vs Stream-Time Accuracy

Two report generations are documented here, clearly versioned:

- **PART A (old code)**: Task 14 report for plan `perclass-difficult-classes-validation`. Scope: DTD (Describable Textures), ViT-B/16 backbone, clip_surgery, seeds 1-2 (full-DTD) / 1-3 (5-class subset). Generated 2026-08-06 from the T10/T11/T12/T13 deliverables. **Verdict: INCONCLUSIVE (delta_mean=5.02 pp, rho_mean=-0.44)**, computed on the BUGGY config path (single-gate, alpha-capped patch term). Preserved as history; corrected and partly superseded by Part B.
- **PART B (follow-up revalidation, fixed code, deterministic loader, seed 1, records_v2)**: Task 10 report for plan `patch-proto-revalidation`. STOP branch: the Wave 2 subset gate FAILED, so Wave 3 was never run. The follow-up established that the patch term is at-best-indistinguishable-from and possibly worse than PTA on the closed-set subset.

All claims are numbered to their evidence file; every number below is traceable to one of the sources in the Sources sections (Part A: S1-S13; Part B: S14-S23).

---

# PART A: Prior plan, old code (preserved history)

## 1. Hypothesis verdict

**Hypothesis (pre-registered):** *as you build better patch-level prototypes over time, classification performance increases* [S9: .omo/plans/perclass-difficult-classes-validation.md:5; S1: outputs/hypothesis_metrics.md:5].

**Decision rule (PatchModPTA-CS full-DTD records only, ablations excluded)** [S1: outputs/hypothesis_metrics.md:7-11]:

- SUPPORT if `delta_mean > +1.0 pp` **AND** `rho_mean > 0.2`
- REFUTE if `delta_mean < 0`
- INCONCLUSIVE otherwise

`acc_first` / `acc_last` = mean(correct) over the first / last 50% of the stream; `delta = acc_last - acc_first` (pp). `rho` = Spearman(cluster growth vs last-half per-class accuracy), masked to classes with nonzero clusters at stream end [S1:13-14].

### Verdict (verbatim from T12, no reinterpretation)

> **DECISION: INCONCLUSIVE (delta_mean=5.02 pp, rho_mean=-0.44)**

[S1: outputs/hypothesis_metrics.md:596; S2: .omo/evidence/task-12-hypothesis.txt:9, 14]. Hand-verified by the orchestrator against raw JSONL: both seeds aggregated, per-seed deltas and rhos match the script exactly [S2:16-40].

### Per-run stream metrics

| label | method | seed | n | acc_first (%) | acc_last (%) | delta (pp) | rho |
|---|---|---|---|---|---|---|---|
| PTA-CS-s1 | PTA | 1 | 1692 | 44.68 | 49.29 | 4.61 | 0.00 |
| PTA-CS-s2 | PTA | 2 | 1692 | 44.33 | 48.46 | 4.14 | 0.00 |
| PatchModPTA-CS-s1 | PatchModPTA | 1 | 1692 | 45.63 | 51.06 | **5.44** | **-0.51** |
| PatchModPTA-CS-s2 | PatchModPTA | 2 | 1692 | 45.39 | 50.00 | **4.61** | **-0.37** |
| ZeroShot-CS-s1 | ZeroShot | 1 | 1692 | 43.38 | 45.39 | 2.01 | 0.00 |
| ZeroShot-CS-s2 | ZeroShot | 2 | 1692 | 43.26 | 45.51 | 2.25 | 0.00 |

[S1:18-25].

### Aggregate and per-seed breakdown (PatchModPTA-CS, decision scope)

- delta_mean = **+5.02 pp** [S1:589]
- rho_mean = **-0.44** [S1:590]
- s1: delta = **+5.44 pp**, rho = **-0.51** [S1:593; S2:27, 30]
- s2: delta = **+4.61 pp**, rho = **-0.37** [S1:594; S2:28, 31]
- delta_mean = (5.44 + 4.61) / 2 = 5.02; rho_mean = (-0.51 + -0.37) / 2 = -0.44. Both seeds included, no cherry-picking [S2:35-40].

### Rolling-window accuracy summary (win=100, non-overlapping; PatchModPTA-CS mean over seeds)

Mean of the per-seed tables at [S1:75-117].

| samples | PatchModPTA-CS-s1 (%) | PatchModPTA-CS-s2 (%) | mean (%) |
|---|---|---|---|
| [0:100) | 45.00 | 43.00 | 44.00 |
| [100:200) | 37.00 | 52.00 | 44.50 |
| [200:300) | 54.00 | 47.00 | 50.50 |
| [300:400) | 55.00 | 39.00 | 47.00 |
| [400:500) | 36.00 | 41.00 | 38.50 |
| [500:600) | 43.00 | 49.00 | 46.00 |
| [600:700) | 44.00 | 53.00 | 48.50 |
| [700:800) | 46.00 | 41.00 | 43.50 |
| [800:900) | 49.00 | 44.00 | 46.50 |
| [900:1000) | 62.00 | 51.00 | 56.50 |
| [1000:1100) | 39.00 | 48.00 | 43.50 |
| [1100:1200) | 50.00 | 43.00 | 46.50 |
| [1200:1300) | 50.00 | 60.00 | 55.00 |
| [1300:1400) | 54.00 | 50.00 | 52.00 |
| [1400:1500) | 46.00 | 45.00 | 45.50 |
| [1500:1600) | 58.00 | 49.00 | 53.50 |
| [1600:1692) | 54.35 | 56.52 | 55.44 |

The windowed curve is noisy (windows bounce between ~38% and ~56%) but the trend is upward: the first two windows average 44.2%, the last two average 54.5%, and `acc_last` exceeds `acc_first` in both seeds (51.06 vs 45.63; 50.00 vs 45.39) [S1:22-23].

### The surprising finding (state it plainly)

**Accuracy improves across the stream, but the improvement is NOT driven by patch-prototype growth.**

- `delta` is strongly **positive**: mean +5.02 pp, i.e. PatchModPTA-CS first-half ~45.5% to last-half ~50.5% [S1:22-23, 589; S2:53-54]. The fear that accuracy would decay over the stream did not materialize.
- `rho` is strongly **negative**: mean -0.44. Classes whose cluster count grows most end up with **lower** last-half accuracy, the opposite sign of the hypothesis [S2:56-58].
- The INCONCLUSIVE verdict is driven by the **rho leg failing**, not by a flat delta. SUPPORT required BOTH conditions (delta > +1.0 pp AND rho > 0.2); the delta leg is TRUE (+5.02), the rho leg is FALSE (-0.44). REFUTE (delta < 0) is FALSE. Rule trace: [S1:7-11; S2:52-60].
- The naive hypothesis ("better patch prototypes -> more clusters -> higher accuracy") is therefore **NOT supported** [S2:55-60]. Something else (warm-up effects, image-level PTA update, stream-order effects, see Caveats) must explain the accuracy gain.

---

## 2. Mechanism analysis

> **SUPERSEDED by Part B (follow-up revalidation, fixed code)**. The two config-read bugs in §2.1-2.2 were FIXED in follow-up Task 1: `multi_gate` now reads nested-first (`models/patch_modulated_pta.py:168`), and the `proto_alpha` machinery was removed entirely. The §2.3 "gate starvation" story is **update-side only**: the quality gate never scales the patch term inside `ProtoAlphaFusion` (`models/fusion.py:123-138`; fusion type set at `configs/base.yaml:39`). See Part B §B.1 for the corrected mechanism narrative and Part B §B.2 for the STOP-branch verdict.

The effective configuration that produced the T10/T12/T13 numbers differs from the intended configuration. Two config-read bugs, both in `models/patch_modulated_pta.py`, make the runs single-gate with a structurally capped patch term [S9:40; S5: models/patch_modulated_pta.py].

### 2.1 Config-read bug 1: `multi_gate` never takes effect

- `models/patch_modulated_pta.py:171` reads `multi_gate = bool(self.cfg.get("multi_gate", False))` at the **top level**, default `False` [S5].
- The intended value lives **nested**: `configs/patch_modulated_pta/patch_modulated_pta.yaml:8` sets `patch_level.multi_gate: true` [S8: configs/patch_modulated_pta/patch_modulated_pta.yaml:8].
- The nested dict is only used for `conf_threshold` / `conf_margin_threshold` (lines 166-167), not for `multi_gate` [S5]. Result: the effective gate is **single-gate** (top-1, conf > 0.3, margin >= 0.00; those two thresholds ARE read correctly from the nested config, patch_modulated_pta.yaml:9-10) [S8:9-10].
- Confirmed by the records themselves: every PatchModPTA-CS record line carries `"gate_mode": "single"` [S10: outputs/records/PatchModPTA-CS-s1/records.jsonl]. The T8 record wiring sets `gate_mode="multi" if multi_gate else "single"` [S5:323], and since the top-level read always yields False, the recorded mode is always "single".

### 2.2 Config-read bug 2: `proto_alpha_max` never takes effect

- `models/patch_modulated_pta.py:169` reads `alpha_max = float(self.cfg.get("proto_alpha_max", 0.2))` at the **top level**, default **0.2** [S5].
- The intended value is nested: `configs/base.yaml` (inherited via `defaults: - ../base` at patch_modulated_pta.yaml:3-4) sets `patch_level.proto_alpha_max: 1.0`, which the top-level read never sees [S8: configs/base.yaml; S9:40].
- Effect: the patch logit term is scaled by `alpha_max * tau_patch_proto` = 0.2 * 10.0 = **2.0** (tau_patch_proto: 10.0 at patch_modulated_pta.yaml:15), while the image-prototype term is weighted by `tau_image_proto` = **80.0** (patch_modulated_pta.yaml:14) [S8:14-15]. The patch term is therefore structurally capped at ~2.0 against an 80.0 image term, i.e. near-negligible by construction [S9:40].

### 2.3 Gate dynamics: difficult classes are starved of updates

- Effective gate is single-gate: top-1 confidence above 0.3 and margin above 0.00 [S8:9-10; S9:40].
- The quality gate is computed over **all classes**; because difficult classes sit at the bottom of the confidence distribution, the gate rarely fires for them [S9:40].
- The consequence shows up in the end-of-stream cluster counts: **8 classes have ZERO clusters at stream end** (flecked, grooved, interlaced, lined, pitted, potholed, stratified, zigzagged), and are masked out of the rho computation [S1:215, 261].
- **3 of the 5 selected difficult classes (flecked, lined, pitted) never built a single cluster across the entire stream** [S1:215, 261; S6: outputs/gap_analysis.md:8].
- The 2 selected classes that DID build clusters still sit at the accuracy floor in the last half: lacelike (5 clusters end, 0.00% last-half, s1) [S1:193], lacelike (4 clusters end, 0.00% last-half, s2) [S1:239], bumpy (6 clusters end, 0.00% last-half, s1) [S1:179], bumpy (5 clusters end, 7.14% last-half, s2) [S1:225]. Clusters are being built for classes that already classify well (e.g. bubbly 100.00%, knitted 100.00%, paisley 100.00% last-half, s1) [S1:178, 192, 197], not for the classes that need help.

### 2.4 Cold start

- `proto_stats_min_count = 5`: prototypes are not trusted until 5 reference observations exist [S8: configs/base.yaml; S7: models/patch_level/gaussian_patch.py:199]. Early in the stream there are zero prototypes for most classes, so the patch term is masked [S9:41].
- `_alpha_from_evidence(n_images, n_half=15.0)` returns 0.0 with zero evidence and saturates at 1.0 after 15 images [S7: models/patch_level/base.py:39-46; S9:41]. Combined with the alpha_max cap of 0.2 (section 2.2), the patch contribution stays small for the whole stream even after the evidence schedule saturates.

### 2.5 Flip diagnostics (offline, NOT part of the decision rule)

Full per-class tables at [S1:269-585]. The pattern:

- **ZeroShot**: `patch_flip` = 0.00% for every class (final logits equal CLIP logits by construction); `flip_vs_img` and `patch_only_flip` are all "n/a" (no patch state) [S1:483-533, 535-585].
- **PTA**: `patch_flip` and `flip_vs_img` are nonzero per class (up to 63.89% for stratified s1, 61.11% for potholed s1) but `patch_only_flip` is "n/a" everywhere, since PTA never builds patch prototypes [S1:275-325, 327-377].
- **PatchModPTA**: `patch_flip` reaches up to 66.67% (smeared s2) [S1:469]; `patch_only_flip` is high on classes that HAVE clusters, e.g. 94.29% (stained s2) [S1:472], 92.59% (matted s2) [S1:459], 86.36% (fibrous s1) [S1:394]. On its own the patch term would flip a majority of predictions for many hard classes, yet the fused final prediction only rarely differs (patch_flip is typically 15-50 points below patch_only_flip). The masked ("n/a") rows in the PatchModPTA tables are exactly the 8 zero-cluster classes including flecked, lined, pitted [S1:395, 411, 457, 463].
- `tau_img` = 80.0 for PatchModPTA-CS vs 100.0 for PTA-CS / ZeroShot-CS (per-run config printout) [S1:275, 327, 379, 431, 483, 535].

---

## 3. Subset results (T13: closed-set 5-class difficult subset)

Selected classes (pre-registered bottom-5 by mean PatchModPTA-CS per-class accuracy, script-selected, no hand-picking): **bumpy, flecked, lacelike, lined, pitted** [S6: outputs/gap_analysis.md:3-8; S4: outputs/result_subset.txt; S11: .omo/evidence/task-13-ablations.txt].

### Per-method per-seed accuracy (%) on the 5-class subset

| method | s1 | s2 | s3 | mean |
|---|---|---|---|---|
| PatchModPTA-CS-sub | 52.22 | 50.00 | 48.89 | **50.37** |
| PTA-CS-sub | 50.56 | 51.67 | 48.89 | **50.37** |
| ZeroShot-CS-sub | 48.33 | 48.33 | 48.33 | **48.33** |

[S4: outputs/result_subset.txt:1-9].

### Reading of the numbers

- **Subset accuracy is HIGHER than full-DTD, not lower.** PatchModPTA-sub mean 50.37 vs full-DTD PatchModPTA-CS-s1 48.35 (and s2 47.70) [S3: outputs/result_perclass.txt:3-4]. This is a **DTD-scoped confound, NOT a claim about adaptation strength**: the 5-class closed-set softmax raises confidences (fewer classes to compete against), so the quality gate fires more often and the update loop sees more high-confidence samples [S9:52, 1138; S12: .omo/notepads/.../learnings.md:278]. The subset is a difficulty floor dev set, not a benchmark [S12:278].
- **PatchModPTA does NOT separate from PTA on the difficult subset**: identical means 50.37 = 50.37 [S4; S12:278]. It beats ZeroShot by +2.04 pp.
- **The 5 selected classes are at the accuracy floor**: 4 of 5 (flecked, lacelike, lined, pitted) are at 0.00% mean accuracy for EVERY method and seed on full-DTD; only bumpy shows any signal (PatchModPTA 1.39 vs PTA 0.00 vs ZeroShot 2.78) [S6:14-18, 22-28, 32-40]. The methods differentiate almost nowhere on this subset because there is essentially nothing to win.

---

## 4. Ablations (T10: single-seed, full-DTD)

| label | override (top-level) | full-DTD acc (%) | delta vs baseline (pp) |
|---|---|---|---|
| PatchModPTA-CS-s1 (baseline) | none | 48.35 | 0.00 |
| PatchModPTA-CS-alpha10-s1 | `--override proto_alpha_max=1.0` | 45.45 | **-2.90** |
| PatchModPTA-CS-multigate-s1 | `--override multi_gate=1` | 48.64 | **+0.29** |

[S3: outputs/result_perclass.txt:4-6; S11: .omo/evidence/task-13-ablations.txt:10-16].

- **Raising the patch-term cap HURTS**: forcing `proto_alpha_max` to 1.0 (the value the nested config intended but the code never read) raises the effective patch scale from 2.0 to 10.0 (0.2*10 -> 1.0*10) and costs **-2.90 pp** [S11:15; S8:15].
- **Multi-gate marginally helps**: `multi_gate=1` gains **+0.29 pp** over the single-gate baseline [S11:16].
- Both ablation records are present (1693 lines each, parity-comparable) [S11:4-8].

---

## 5. Decision inputs for the next step

Context: the user's step 4 is "possibly build a proper way of using patch-based prototypes". This section states what the metrics imply for that work. **Implementation is future work and OUT of scope for this plan** [S9:1149].

1. **The naive patch-prototype hypothesis is falsified as stated.** Cluster growth anti-correlates with accuracy (rho_mean = -0.44 [S1:590]); more clusters did not produce better accuracy. Any future design should not assume "more/better prototypes -> better classification" [S2:55-60].

2. **The patch term has not actually been exercised at its intended strength.** Because of the `proto_alpha_max` read bug, every PatchModPTA-CS run ran with the patch term capped at ~2.0 against an image term of 80 [S9:40; S5:169; S8:14-15]. The only time the cap was raised (alpha10), accuracy went DOWN by 2.90 pp [S11:15]. So "just raise the cap" is the wrong lever; the mechanism needs redesign, not retuning.

3. **The gate starves the classes that need updates.** 3 of the 5 selected difficult classes (flecked, lined, pitted) never built a single cluster in either seed [S1:215, 261], and the classes that did build clusters (lacelike, bumpy) still classify at ~0% last-half [S1:179, 193, 225, 239]. A proper design would condition prototype building and updating on per-class difficulty or accuracy, not only on top-1 confidence and margin [S9:1143].

4. **Single-gate vs multi-gate is worth a real comparison, but only after the config-read is fixed.** The multigate ablation gained only +0.29 pp while running through the buggy top-level read path; a faithful multi-gate evaluation needs the config to actually reach the code [S11:16; S5:171].

5. **Ground truth for re-evaluation already exists.** Parity-validated records (48.35 == 48.35, 46.99 == 46.99 [S13: .omo/evidence/task-10-parity-gate.txt:7, 14]) mean any redesigned prototype mechanism can be benchmarked against the same recordings without re-running the baseline.

---

## 6. Caveats

1. **DTD-scoped.** All results are on Describable Textures only. No claim is made for other datasets [S1; S9:1150].
2. **Limited seeds.** 2 seeds for full-DTD (PatchModPTA-CS-s1/s2), 3 seeds for the subset [S9:35; S12:275-277].
3. **Effective config is single-gate.** The multi_gate config bug means all runs used the single-gate path; the intended multi-gate behavior was never executed [S5:171; S9:40; S10].
4. **5-class closed-set confound.** The subset run remaps labels to 0..K-1 and builds text embeddings from subset classnames only; elevated confidences inflate gate firing. The higher subset accuracy vs full-DTD is a measurement artifact of the closed-set setup, not evidence of adaptation strength [S9:52, 1138; S12:278].
5. **Loader batch order is nondeterministic.** `build_data_loader` uses `shuffle=True, num_workers=8`, so two runs with the same seed see different stream orders (probe-verified in T7) [S12:204-210]. Stream-order metrics (delta, rolling windows) carry this noise; the parity gate happened to match exactly this time [S13:7, 14].
6. **rho is a snapshot correlation, not a causal test.** It correlates end-of-stream cluster count with last-half accuracy; it does not establish direction of causality [S1:14].
7. **Recording is behavior-neutral.** Parity PASS quoted: PatchModPTA-CS-s1 48.35 == ref 48.35; PTA-CS-s1 46.99 == ref 46.99 [S13:7, 14]. The records are therefore trustworthy for the analyses above.
8. **Flip diagnostics are offline and excluded from the decision rule** [S1:269].

---

## Sources

- [S1] `outputs/hypothesis_metrics.md` (T12, 596 lines) — verdict, per-run metrics, rolling windows, rho tables, flip diagnostics.
- [S2] `.omo/evidence/task-12-hypothesis.txt` (T12) — verdict output, hand cross-checks, seed aggregation check.
- [S3] `outputs/result_perclass.txt` (T10, 8 lines) — full-DTD + ablation accuracies.
- [S4] `outputs/result_subset.txt` (T13, 9 lines) — subset accuracies.
- [S5] `models/patch_modulated_pta.py` — config-read bug sites (lines 169, 171), gate_mode recording (line 323).
- [S6] `outputs/gap_analysis.md` + `outputs/perclass_tables.md` (T11) — class selection and per-class gaps.
- [S7] `models/patch_level/base.py` (lines 39-46), `models/patch_level/gaussian_patch.py` (line 199) — evidence schedule, stats min count.
- [S8] `configs/patch_modulated_pta/patch_modulated_pta.yaml`, `configs/base.yaml` — nested config values (multi_gate: true, conf 0.3 / 0.00, proto_alpha_max: 1.0, tau_image_proto 80, tau_patch_proto 10).
- [S9] `.omo/plans/perclass-difficult-classes-validation.md` — hypothesis, research findings, Task 14 spec, gate dynamics.
- [S10] `outputs/records/PatchModPTA-CS-s1/records.jsonl` — `"gate_mode": "single"` per sample.
- [S11] `.omo/evidence/task-13-ablations.txt` — ablation records + deltas.
- [S12] `.omo/notepads/perclass-difficult-classes-validation/learnings.md` — T7 loader determinism, T10-T13 outcomes.
- [S13] `.omo/evidence/task-10-parity-gate.txt` — parity PASS values.

---

# PART B: Follow-up revalidation (fixed code, deterministic loader, seed 1, records_v2)

Task 10 report for plan `patch-proto-revalidation`. Generated 2026-08-06 from the Task 1/2/3/5/6 deliverables. This part supersedes the Part A mechanism narrative (§2) and the Part A verdict context (§1) with corrected, fixed-code information. Every number traces to evidence sources S14-S23.

## B.1 Corrected mechanism narrative

1. **`quality_gate` is DISCARDED inside the fusion; the prior plan's §2.3 "gate starvation" story is update-side only.** The effective fusion type is `ProtoAlphaFusion` (`configs/base.yaml:39`), whose forward (`models/fusion.py:123-138`) computes `tau_text * clip + tau_image_proto * image_proto + (tau_patch_proto * proto_alpha * patch_proto if patch_proto is not None)`. There is **no `quality_gate` term** in that sum. Only `QualityGatedFusion` (`models/fusion.py:141-152`) would multiply the patch term by `quality_gate`, and it is not the configured type. The `quality_gate` computed at `models/patch_level/gaussian_patch.py:307` is consumed at `patch_modulated_pta.py:89-90` as a scalar on the **image-EMA update rate** only: `w_new[mask] *= (1 + quality_modulation * quality_gate)`. Gate starvation can therefore suppress image-prototype EMA updates for hard classes, but it NEVER gates the patch term. The Part A §2.3 claim that the gate "structurally caps" the patch term is mechanism-side incorrect; the corrected reading is update-side only.
2. **`multi_gate` is now actually effective.** Follow-up Task 1 fixed the read: `multi_gate = bool(_pl_cfg.get("multi_gate", self.cfg.get("multi_gate", False)))` at `models/patch_modulated_pta.py:168`, nested-first, matching `patch_level.multi_gate: true` at `configs/base.yaml:16` [S16]. All three PatchModPTA-fixed runs carry `"gate_mode": "multi"` on 180/180 records [S18].
3. **`proto_alpha` removed entirely.** The Task 1 diff deleted the `proto_alpha_max` read, the `n_half` evidence schedule, and the `_alpha_from_evidence` call; the fusion default `proto_alpha=1.0` is now constant (`models/fusion.py:119, 133`) [S16]. The patch term is therefore `tau_patch_proto * patch_proto` wherever clusters exist; the `proto_stats_min_count: 5` cluster-existence guard is preserved (`configs/base.yaml:32`) [S16]. Records show `proto_alpha==1.0` on 180/180 [S18; S14:8].
4. **`aggregation` default is "zscore".** `models/patch_level/gaussian_patch.py:196`: `aggregation = str(self._cfg.get("aggregation", "zscore"))`. The comment at `configs/base.yaml:24-26` claiming "the code default (top_m_mean)" is STALE; the actual default is `zscore`. This write-up corrects the comment.
5. **`tau_img` 80 vs 100 is intentional, not a bug.** `configs/patch_modulated_pta/patch_modulated_pta.yaml:15` sets `tau_image_proto: 80.0`; `configs/PTA/pta.yaml:7` sets `100.0`. PatchModPTA fuses at 80, PTA/ZeroShot at 100 (flip table headers confirm: `tau_img = 80.0` for the three PMP runs, `100.0` for PTA/ZeroShot) [S22; S23].
6. **Audit correction: only two dead reads remain.** The plan assumed `conf_source` and `seed` were dead reads; the audit verified them LIVE: `conf_source` used at `patch_modulated_pta.py:273-282` (logits branch), `seed` at `:199/:367` (record header + summary) [S16:36-37, 41]. Only `max_K` (:162) and `match_threshold` (:163) are dead-read-untouched (single occurrence, never consumed) [S16:31-32, 70-73].

## B.2 STOP-branch verdict: Wave 2 subset gate FAILED

Pre-registered gate (LOCKED before any Wave 2 record existed, applied mechanically) [S15: §5]:

```
proceed iff max over tau of (PatchModPTA-fixed-sub mean) >= PTA-fixed-sub mean
```

Wave 2 subset means (seed 1, 180 records each, fixed code, deterministic loader) [S14: §1; S18]:

| run | method | tau | n | correct | subset mean acc (%) |
|---|---|---|---|---|---|
| PatchModPTA-fixed-tau2.5-s1 | PatchModPTA-fixed | 2.5 | 180 | 90 | **50.00** |
| PatchModPTA-fixed-tau5.0-s1 | PatchModPTA-fixed | 5.0 | 180 | 85 | **47.22** |
| PatchModPTA-fixed-tau10-s1 | PatchModPTA-fixed | 10.0 | 180 | 79 | **43.89** |
| PTA-fixed-s1 | PTA-fixed | n/a | 180 | 93 | **51.67** |
| ZeroShot-fixed-s1 | ZeroShot-fixed | n/a | 180 | 87 | **48.33** |

- `max_tau PatchModPTA-fixed-sub` = max(50.00, 47.22, 43.89) = **50.00** (at tau 2.5)
- `PTA-fixed-sub` = **51.67**
- 50.00 < 51.67, difference **-1.67 pp**, not a tie, so the tie→proceed rule is NOT invoked [S14:34-36]
- **GATE VERDICT: FAIL → STOP branch.** Wave 3 (Tasks 7/8/9: full-DTD per-seed runs, point-5 bar, decide()) was NOT run, per the pre-registered rule [S14:38-40, 86-88; S18:36-39].

Sweep sanity: the tau sweep has signal, not noise. PatchModPTA-fixed-sub is strictly monotonic decreasing in tau (50.00 → 47.22 → 43.89, spread 6.11 pp) [S14:24; S17:47-51].

**The point-5 bar was NOT tested.** The pre-registered point-5 bar (`rho_mean > 0 AND delta_mean > 0`) is defined on full-DTD per-seed PatchModPTA stream metrics (delta = last-half minus first-half accuracy; rho = Spearman of cluster growth vs last-half per-class accuracy). Only Wave 3 would produce those runs, and the STOP branch precluded Wave 3. The subset means above are a gate instrument, not a point-5 substitute: rho/delta cannot be computed from subset records (single class stream per subset sample, no cluster-growth window comparable to full-DTD). Do not read the subset numbers as a point-5 verdict.

**This SUPERSEDES the old INCONCLUSIVE verdict context** (Part A §1, preserved above with its old-code label). The old verdict (delta_mean=+5.02 pp, rho_mean=-0.44) was computed on old-code full-DTD records whose effective config was single-gate with an alpha-capped patch term (Part A §2). The follow-up gate failure concerns fixed-code subset runs. The two cannot be compared directly; the follow-up's conclusion is narrower: on the fixed path, with the patch term actually exercised at its intended scale, the patch term is at-best-indistinguishable-from and possibly worse than PTA on the closed-set subset.

## B.3 Per-class gate breakdown + cluster context

Per-class, PatchModPTA-fixed-tau2.5 vs PTA-fixed, seed 1 [S14: §3.1]:

| class | PatchModPTA tau2.5 (%) | PTA (%) | ZeroShot (%) | Δ vs PTA (pp) | verdict |
|---|---|---|---|---|---|
| bumpy | 72.2 (26/36) | 80.6 (29/36) | 66.7 (24/36) | **-8.3** | worsened |
| flecked | 0.0 (0/36) | 2.8 (1/36) | 0.0 (0/36) | **-2.8** | worsened (near-floor for all methods) |
| lacelike | 69.4 (25/36) | 75.0 (27/36) | 77.8 (28/36) | **-5.6** | worsened |
| lined | 91.7 (33/36) | 88.9 (32/36) | 75.0 (27/36) | **+2.8** | improved |
| pitted | 16.7 (6/36) | 11.1 (4/36) | 22.2 (8/36) | **+5.6** | improved |

Full tau sweep per class (Δ vs PTA, pp) [S14: §3.1]:

| class | tau2.5 | tau5.0 | tau10 |
|---|---|---|---|
| bumpy | -8.3 | -8.3 | -8.3 |
| flecked | -2.8 | -2.8 | -2.8 |
| lacelike | -5.6 | -22.2 | -36.1 |
| lined | +2.8 | +5.6 | +0.0 |
| pitted | +5.6 | +5.6 | +8.3 |

**Summary**: 3 classes worsened (bumpy, flecked, lacelike), 2 improved (lined, pitted). lacelike collapses monotonically with tau (-5.6 → -22.2 → -36.1 pp). The lined/pitted gains (+2.8…+8.3 pp) do not offset the lacelike collapse plus the flat -8.3 pp on bumpy. The subset mean stays below PTA at every tau [S14:68].

**Cluster context** (true-side `proto_stats`, n=36 per class) [S19; S14: §3.2]:

| run | bumpy | flecked | lacelike | lined | pitted | classes fully zero |
|---|---|---|---|---|---|---|
| PatchModPTA-fixed-tau2.5-s1 | 0/36 | 4/36 | 1/36 | 1/36 | 2/36 | 0 |
| PatchModPTA-fixed-tau5.0-s1 | 0/36 | 4/36 | 1/36 | 1/36 | 2/36 | 0 |
| PatchModPTA-fixed-tau10-s1 | 0/36 | 4/36 | 1/36 | 1/36 | 2/36 | 0 |
| PTA-fixed-s1 | 36/36* | 36/36* | 36/36* | 36/36* | 36/36* | 5 |
| ZeroShot-fixed-s1 | 36/36* | 36/36* | 36/36* | 36/36* | 36/36* | 5 |

\* PTA/ZeroShot `proto_stats` sides are null on all 180 records: the patch term is structurally absent for these methods by design [S19:18-19].

In the PatchModPTA runs every class forms clusters on ≥32 of 36 samples (flecked is the most starved at 4/36 zero, and it is also the hardest class at 0.0-2.8% for every method). **The gate miss (-1.67 pp at best tau) cannot be blamed on missing clusters**: the patch term had full class access and still under-performed PTA [S14:84; S19:23-28].

## B.4 Flip diagnostics (flips-v2 on the NEW subset records, GT-aware)

Command: `python scripts/analyze_records.py flips-v2 --records outputs/records_v2/subset --out /tmp/flip-subset.md` (Task 3 CLI, commit b9c0ccb). Per-component predictions are reconstructed from STORED logits using the ProtoAlphaFusion formula with `proto_alpha = 1.0`; `correction` = previous component wrong → this component right; `regression` = previous component right → this component wrong; flip acc = corr / (corr + reg); net = corr - reg. Patch rows are masked to records with `proto_stats.pred.n_clusters > 0` [S23].

**Class-name mapping caveat**: subset record headers carry `classnames==[]`, so the CLI falls back to full-DTD classname order for display. The subset targets are re-indexed 0-4 in class-file order: class_0=bumpy, class_1=flecked, class_2=lacelike, class_3=lined, class_4=pitted. The names in the tables below are post-processed to the true subset classes. The mapping was verified against the ZeroShot per-class accuracy column in [S14: §3.1] (exact match for all 5 rows: 66.67/0.00/77.78/75.00/22.22 = ZeroShot 66.7/0.0/77.8/75.0/22.2). The rates are unaffected; only the labels were re-mapped [S23].

Per-run component summary [S23]:

| run | text-only acc (%) | +image (corr/reg, acc %, net) | +patch-alone (corr/reg, acc %, net) | +patch-with-image (corr/reg, acc %, net) |
|---|---|---|---|---|
| PTA-fixed-s1 | 48.33 (87/180) | 15/9 (62.50, **+6**) | masked 180 | masked 180 |
| PatchModPTA-fixed-tau2.5-s1 | 48.33 (87/180) | 12/6 (66.67, **+6**) | 17/39 (30.36, **-22**) | 2/6 (25.00, **-4**) |
| PatchModPTA-fixed-tau5.0-s1 | 48.33 (87/180) | 12/6 (66.67, **+6**) | 17/39 (30.36, **-22**) | 6/14 (30.00, **-8**) |
| PatchModPTA-fixed-tau10-s1 | 48.33 (87/180) | 12/6 (66.67, **+6**) | 18/39 (31.58, **-21**) | 8/22 (26.67, **-14**) |
| ZeroShot-fixed-s1 | 48.33 (87/180) | masked | masked 180 | masked 180 |

Reading of the numbers:

- **text-only = 48.33 (87/180) for every run.** Identical CLIP predictions across all 5 runs is a determinism sanity check: same seed 1, same deterministic stream (Task 2 fix), same text logits. [S23]
- **The image-EMA prototype carries the entire observable gain**: +image lifts accuracy from 48.33 to 62.50 (PTA) or 66.67 (PMP), net +6 corrections in every run. [S23]
- **patch-alone is a net detractor at every tau** (net -21 to -22; flip acc 30-32%): on its own the patch term flips mostly to wrong answers. [S23]
- **patch-with-image is negative at every tau** (net -4/-8/-14 at tau 2.5/5.0/10.0), strictly worsening with tau. The image term pulls the fused prediction partway back, but the patch contribution still subtracts. [S23]

Per-class (cluster-stratified), PatchModPTA-fixed-tau2.5-s1 (tau_img = 80.0, tau_patch_proto = 2.5) [S23]:

| class | n | n_cluster | text-only (%) | +image (corr/reg, net) | +patch-alone (corr/reg, net) | +patch-with-image (corr/reg, net) |
|---|---|---|---|---|---|---|
| bumpy | 36 | 35 | 66.67 | 6/3 (+3) | 6/11 (-5) | 0/1 (-1) |
| flecked | 36 | 36 | 0.00 | 1/0 (+1) | 0/0 (-) | 0/1 (-1) |
| lacelike | 36 | 34 | 77.78 | 1/2 (-1) | 1/20 (-19) | 0/3 (-3) |
| lined | 36 | 35 | 75.00 | 4/0 (+4) | 6/3 (+3) | 2/0 (+2) |
| pitted | 36 | 34 | 22.22 | 0/1 (-1) | 4/5 (-1) | 0/1 (-1) |

Per-class, PatchModPTA-fixed-tau5.0-s1 (tau_img = 80.0, tau_patch_proto = 5.0) [S23]:

| class | n | n_cluster | text-only (%) | +image (corr/reg, net) | +patch-alone (corr/reg, net) | +patch-with-image (corr/reg, net) |
|---|---|---|---|---|---|---|
| bumpy | 36 | 35 | 66.67 | 6/3 (+3) | 6/11 (-5) | 1/2 (-1) |
| flecked | 36 | 36 | 0.00 | 1/0 (+1) | 0/0 (-) | 0/1 (-1) |
| lacelike | 36 | 34 | 77.78 | 1/2 (-1) | 1/20 (-19) | 1/9 (-8) |
| lined | 36 | 35 | 75.00 | 4/0 (+4) | 6/3 (+3) | 3/0 (+3) |
| pitted | 36 | 34 | 22.22 | 0/1 (-1) | 4/5 (-1) | 1/2 (-1) |

Per-class, PatchModPTA-fixed-tau10-s1 (tau_img = 80.0, tau_patch_proto = 10.0) [S23]:

| class | n | n_cluster | text-only (%) | +image (corr/reg, net) | +patch-alone (corr/reg, net) | +patch-with-image (corr/reg, net) |
|---|---|---|---|---|---|---|
| bumpy | 36 | 35 | 66.67 | 6/3 (+3) | 6/11 (-5) | 1/2 (-1) |
| flecked | 36 | 36 | 0.00 | 1/0 (+1) | 0/0 (-) | 0/1 (-1) |
| lacelike | 36 | 35 | 77.78 | 1/2 (-1) | 2/20 (-18) | 2/15 (-13) |
| lined | 36 | 35 | 75.00 | 4/0 (+4) | 6/3 (+3) | 3/2 (+1) |
| pitted | 36 | 34 | 22.22 | 0/1 (-1) | 4/5 (-1) | 2/2 (+0) |

PTA-fixed-s1 per-class (patch rows are all "patch contribution N/A", no patch clusters by design) [S23]:

| class | n | n_cluster | text-only (%) | +image (corr/reg, net) |
|---|---|---|---|---|
| bumpy | 36 | 0 | 66.67 | 7/2 (+5) |
| flecked | 36 | 0 | 0.00 | 1/0 (+1) |
| lacelike | 36 | 0 | 77.78 | 1/2 (-1) |
| lined | 36 | 0 | 75.00 | 6/1 (+5) |
| pitted | 36 | 0 | 22.22 | 0/4 (-4) |

ZeroShot-fixed-s1 per-class: text-only only (bumpy 66.67, flecked 0.00, lacelike 77.78, lined 75.00, pitted 22.22); image and patch rows are masked (no adaptation state) [S23].

Note on the two cluster views: the flip tables use the pred-side cluster mask (`proto_stats.pred.n_clusters > 0`), which is why flecked shows n_cluster 36/36 (the model almost never predicts flecked, so its records' pred-side clusters belong to other classes) while the true-side zero counts in B.3 show flecked at 4/36. The B.3 true-side counts are the class-access numbers; the flip-table mask is a per-record availability filter. Both are reported; neither contradicts the other [S19; S23].

## B.5 What the negative result means for the design

1. **On the fixed code path the patch term is at-best-indistinguishable-from, and possibly worse than, PTA on the closed-set subset.** Best tau (2.5) reaches 50.00 vs PTA 51.67; higher tau is monotonically worse (47.22, 43.89) [S14].
2. **The flip diagnostics locate the damage.** The image-EMA prototype alone delivers the full observed gain (+6 net corrections); the patch-alone component is a net detractor (-21 to -22), and the fused patch contribution scales harmfully with tau (-4/-8/-14) [S23].
3. **lacelike is the canary**: high text-only accuracy (77.78%) that the patch term systematically destroys (patch-alone net -19/-19/-18 across tau; fused net -3/-8/-13). A patch prototype for a class the model already classifies well adds noise, not signal [S23].
4. **Old-code parity is broken by the fix.** Part A §3 reported PatchModPTA 50.37 = PTA 50.37 on the buggy path; on the fixed path (multi_gate effective, alpha cap removed) the patch term loses parity with PTA. The prior multigate ablation (+0.29 pp) and the alpha10 ablation (-2.90 pp) were measured through the buggy path and are not directly comparable [S14:97].
5. **The prior plan's "just raise the cap" conclusion (Part A §5.2) is now moot.** The cap no longer exists (proto_alpha = 1.0 constant), and the uncapped patch term is worse, not better [S16].
6. **Design implication**: any redesigned patch-prototype mechanism must (a) stop the patch term from degrading already-easy classes such as lacelike, and (b) justify itself against the image-EMA alone, which already produces the full observed gain. The hypothesis "as you build better patch-level prototypes, classification improves" remains unsupported; the corrected mechanism does not rescue it on this subset [S14; S23].

## B.6 Sources (follow-up)

- [S14] `.omo/evidence/wave2-gate.md`: verdict source, per-run means (§1), gate check (§2), per-class table + cluster context (§3.1-3.2), top-2 N/A (§3.3).
- [S15] `.omo/evidence/gate-noise-sizing.md`: pre-registered Wave 2 gate procedure (§5), single-seed noise floor (§3), old-data sensitivity (§4).
- [S16] `.omo/evidence/config-read-audit.md`: 17-read audit; multi_gate fixed(T1); proto_alpha removed; conf_source/seed LIVE; max_K/match_threshold dead-read-untouched.
- [S17] `.omo/evidence/task-5-sweep-attest.txt`: sweep attestation, tau resolution, means, red-flag check.
- [S18] `.omo/evidence/task-6-gate-check.txt`: reproducible verdict (50.00/47.22/43.89/51.67/48.33), fixed-code fingerprint (gate_mode multi, proto_alpha 1.0).
- [S19] `.omo/evidence/task-6-cluster-count.txt`: true-side zero-cluster counts per class.
- [S20] `models/fusion.py` (ProtoAlphaFusion.forward :123-138, QualityGatedFusion :141-152): quality_gate discarded inside ProtoAlphaFusion; proto_alpha default 1.0.
- [S21] `models/patch_modulated_pta.py` (:89-90 quality modulation on image-EMA, :168 multi_gate nested read, :167 conf_source, :191 seed, :162-163 dead reads), `models/patch_level/gaussian_patch.py` (:196 aggregation default "zscore", :307 quality_gate), `configs/base.yaml` (:16 multi_gate, :39 fusion type, :24-26 stale comment).
- [S22] `configs/patch_modulated_pta/patch_modulated_pta.yaml` (:15 tau_image_proto 80.0), `configs/PTA/pta.yaml` (:7 tau_image_proto 100.0).
- [S23] `/tmp/flip-subset.md`: flips-v2 output on `outputs/records_v2/subset` (GT-aware flip metrics; class names post-processed per B.4 note).
