# Hypothesis Validation Findings — Patch-Prototype Growth vs Stream-Time Accuracy

Task 14 report for plan `perclass-difficult-classes-validation`.
Scope: DTD (Describable Textures), ViT-B/16 backbone, clip_surgery, seeds 1-2 (full-DTD) / 1-3 (5-class subset).
Generated: 2026-08-06, from the T10/T11/T12/T13 deliverables. All claims are numbered to their evidence file; every number below is traceable to one of the sources in the Sources section.

---

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
