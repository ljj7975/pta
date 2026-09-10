# Experimental Results

This directory consolidates the experimental findings from the PTA improvement investigation. It covers three bodies of work: (1) the patch-level fusion study, (2) the confusability/repulsion line (Phases 1–10: read-time fusion, write-time gates/reweighting, prototype repulsion), and (3) the representation-upgrade study (waves 1–3).

## Documents

### [PTA_Limitations_and_Patch_Signal_Analysis.md](./PTA_Limitations_and_Patch_Signal_Analysis.md)

The patch-level fusion investigation — a self-contained report covering:

1. **Background** — How PTA works, image-level vs. patch-level prototypes, write-time vs. inference-time behavior
2. **PTA's Prototype Drift Problem** — Why image-level prototypes fail when CLIP is confidently wrong
3. **Patch-Level Prototype Investigation** — Why patch fusion doesn't fix PTA (bank quality, separability, structural asymmetry)
4. **Patch Agreement as Quality Signal** — The validated meta-signal and why it doesn't convert to accuracy wins
5. **Fusion Mechanisms** — How different fusion mechanisms perform and why always-on fusion collapses
6. **Summary and Supplementary Data**

### [METHODOLOGY.md](./METHODOLOGY.md)

Detailed description of how each finding was derived, including:
- Exact script used (under `scripts/` folder)
- Input data and write-time settings
- Methodology and computation details
- Output file containing the numbers

### [PatchModPTA_Purity_Separability_Trust_Analysis.md](./PatchModPTA_Purity_Separability_Trust_Analysis.md)

Follow-up to the report above: re-runs the patch-modulation investigation from clean checkpoints and adds
the write-time trust-lever studies (Parts 4a/4b) — hard write-gating and write-weight boosting/down-
weighting on the confidence x patch-agreement signal. Both fail to beat base PTA.

### [Phase1-3_Read_Time_Embedding_and_MultiProto_Null_Results.md](./Phase1-3_Read_Time_Embedding_and_MultiProto_Null_Results.md)

Three further, previously-untried directions, each tested to a pre-registered go/no-go rule:
1. **Read-time trust-adaptive fusion** — full sweep (120 runs); no-go, with a clean mechanistic
   explanation (regime/tie overlap analysis) for why the signal helps nowhere on the read side either.
2. **Foreground-weighted embedding** — killed at the cheap-diagnostic stage (CLIP-Surgery-weighted patch
   pooling collapses -40 to -44pp vs. CLS on fine-grained datasets).
3. **Image-level multi-prototype** — killed at the cheap-diagnostic stage (clustering margins near zero,
   not positive, despite being far better than the patch-level -0.11 to -0.24).

### [Phase4-6_Calibration_Confusability_ViewConsistency_Null_Results.md](./Phase4-6_Calibration_Confusability_ViewConsistency_Null_Results.md)

Three directions chosen to share nothing with patch content or patch-vote signals, each tested to a
pre-registered go/no-go rule:
1. **Class-prediction-frequency calibration** — offline replay of existing records (no new GPU runs);
   no-go, peak effect +0.15pp and reverses sign on higher-class-count datasets.
2. **Prototype confusability monitor** — the diagnostic is the strongest positive signal in the campaign
   (+9 to +16.5pp tercile accuracy gap from the bank's own geometry, no CLIP confidence or patch content
   involved), but the write-freeze adapter built on it regresses monotonically and severely (-2.2 to
   -10.0pp) — freezing a class starves it of the good writes it needs, not just the bad ones.
3. **Multi-view consistency trust signal** — the diagnostic is even stronger (+16.6 to +27.4pp purity gap,
   comparable to or exceeding the patch-vote signal), but plugged into the same read-time reweighting
   lever from direction 1 of the prior document, it produces the same null result — cross-tabulation shows
   this trust signal overlaps with "tie" samples almost identically to the unrelated patch-vote signal
   (~88% of ties are "untrusted" in both), generalizing the earlier finding: read-time reweighting is
   structurally limited regardless of the trust signal's source or quality.

### [Phase7_WriteTime_TrustGate_Reweight_Null_Results.md](./Phase7_WriteTime_TrustGate_Reweight_Null_Results.md)

Takes the two strong diagnostics from the prior document (multi-view consistency, prototype
confusability) and, for the first time, plugs each into the write-time hard-gate and two-sided reweight
levers from `PatchModPTA_Purity_Separability_Trust_Analysis.md` (Parts 4a/4b) — independently and
combined (AND'd together) — instead of patch-vote. All 18 non-control settings (7 gate + 13 reweight) are
no-go. Confusability is consistently the mildest of the three signals and removing its earlier
permanence substantially narrows the loss (−0.8 to −2.1pp vs. the frozen version's −2.2 to −10.0pp), but
never flips positive; multi-view is the most destructive write-time lever in the whole campaign despite
being the single strongest standalone diagnostic. Cross-cutting conclusion: diagnostic strength is
inversely related to write-time usefulness across all three signals tested so far, and combining signals
never beats the stronger individual one.

### [Phase8_Prototype_Repulsion_Results.md](./Phase8_Prototype_Repulsion_Results.md)

Tests a mechanistically different family from every prior write-time study: instead of gating or
reweighting the write, the write always fires unmodified and an explicit repulsion step pushes two
confusable class prototypes apart afterward. All 6 settings are no-go by the standard bar (`oxford_pets`
regresses monotonically at every dose, down to −8.0pp), but the mechanism check confirms the correction
does exactly what it claims — it measurably reduces nearest-other-class confusability every time it fires
— and the lowest dose (`lr=0.02`, confusable-triggered) is the only per-sample correction in the whole
confusability line of investigation to post a *positive* delta on two of three dev datasets. Root cause:
the confusability signal can't distinguish "close because of drift" from "close because of genuine
fine-grained visual similarity" (oxford_pets breeds), and repulsion actively damages the latter case
instead of merely under-serving it the way gating did.

### [Phase9_Drift_Gated_Repulsion_Results.md](./Phase9_Drift_Gated_Repulsion_Results.md)

Follow-up to Phase 8: adds a drift-velocity signal (how much a class's own prototype direction has moved
recently) to distinguish drift-confusable pairs (repulsion is a legitimate fix) from genuinely-similar
pairs (repulsion just distorts a good boundary), plus a repulsion floor to stop unbounded compounding.
Still no-go by the standard bar, but the failure mode is far milder than Phase 8: no catastrophic
collapse anywhere, `dtd`/`oxford_flowers` positive in every setting, and `lr=0.02` has zero regression
>0.5pp on any dataset — the cleanest result in the whole confusability-repulsion line. The drift-gating
hypothesis is directionally confirmed (`drift_confusable` beats a `stable_confusable` counterfactual on
`oxford_pets` at every dose, holding the floor fixed), but most of the recovery from Phase 8's damage
comes from the floor and from firing far less often in general, not from the drift signal specifically —
an unresolved ablation (floor alone, no drift gate) is flagged as the natural next step.

### [Phase10_Drift_Floor_Anchor_Diagnostics_Results.md](./Phase10_Drift_Floor_Anchor_Diagnostics_Results.md)

Four diagnostic experiments resolving open questions from the Phases 8-9 confusability-repulsion
line: (1) a floor-ablation 2x2 decomposition proving the drift gate, not the floor, drives Phase 9's
recovery (drift contributes ~100% of the oxford_pets improvement at lr=0.02, floor ~0%); (2)
text-anchored EMA, no-go (best avg +0.046pp, non-monotonic, early convergence degrades); (3)
probability-weighted EMA, no-go (p^gamma downweighting hurts dtd most, −0.59 to −2.19pp); (4)
bilateral-drift OR-trigger, mixed/no-go (fires 33-46% more than Phase 9's AND-condition but does
not translate to accuracy gains). All 16 non-control settings no-go across all four experiments.
Concludes the confusability-repulsion line of investigation (Phases 8-10) has been exhaustively
explored with no promotable result and recommends closing this line entirely.

### [PTA_Representation_Upgrade_Analysis.md](./PTA_Representation_Upgrade_Analysis.md)

The representation-upgrade study (waves 1–3): does improving PTA's per-class prototype — Gaussian scoring, prototype banks, compactness write gates, augmentation ensembling, text-anchor damping, and their combinations — beat base PTA? Self-contained report with four parts:

- **Part 0: Pre-validation gates** — every mechanism was gated on cheap offline data before any cluster compute (write-purity deciles, Mahalanobis proxy, k-means bank proxy, hijacking check)
- **Part 1: Methods** — GaussPTA (variance-aware Mahalanobis scoring), BankPTA (K-prototype bank), CompactPTA (distance write gate), plus wave-2 BankV2 (anti-collapse bank), BankCompact, GaussCompact and wave-3 AugPTA (entropy-selected view ensembling), AnchorPTA (text-anchor damping), ablations and an α/T sweep
- **Part 2–3: Results** — all methods fail to beat base PTA; each fails for a distinct, documented reason (variance-estimation amplification, rich-get-richer collapse, error-mode growth, gate-removes-correct-updates, ensembling degrades evidence, damping degenerates to uniform scaling)
- **Part 4: Wave 3** — 4-seed accuracy, write-purity analysis, identity ablation (V1K1 == base PTA, lossless GPU reducibility confirmed), sweep showing default α=0.01/T=20 is not a strawman

Bottom line: the prototype is the right lever (it beats CLIP on CLIP↔prototype ties), but no mechanism whose advantage depends on statistics estimated from CLIP's own biased guesses has beaten base PTA. The write-stream bias is the binding constraint.

---

## Key Takeaways — Patch-Level Fusion

1. **PTA has a structural limitation**: when CLIP and the image-level prototype disagree, the prototype is wrong 75.3% of the time — but no confidence-based filter can catch this.

2. **Patch-level evidence provides a genuinely independent signal**: whether CLIP's CLS-level guess and the patch-level vote agree is a strong indicator of prediction correctness (30+ pp purity gap).

3. **This signal does not convert to accuracy wins**: across 30+ experiments, patch content fused into predictions never reliably beats plain PTA.

4. **The fundamental issue is structural**: patch banks are built from CLIP's own guesses (no ground truth), class clusters overlap at the patch level (negative separability margins everywhere), and the signal quality is fixed by the underlying CLIP features.

## Key Takeaways — Representation Upgrade (Waves 1–3)

1. **All prototype-representation upgrades fail to beat base PTA** on dtd / oxford_flowers / oxford_pets (4-seed means): GaussPTA −3.4 to −32.8pp, BankPTA −0.6pp (collapse: K=3 ≡ K=5), BankV2 −4.4 to −5.5pp, CompactPTA −0.2 to −6.0pp, combinations −5.9 to −21.7pp, AugPTA −0.2 to −2.0pp, AnchorPTA −0.01/−0.02pp (ties).

2. **Every offline gate passed, every online method failed — for distinct reasons**: variance estimation from a biased stream amplifies Mahalanobis error (Gauss); nearest-assignment cannot create K-way diversity (Bank v1); threshold-gated growth plants full-strength error modes from wrong writes (BankV2); gating removes correct updates along with wrong ones (Compact); ensembled logits are *less* accurate than the single view (Aug); damping fires on ~all writes after warm-up (Anchor).

3. **The gap between a favorable offline proxy and the label-free online setting is the binding constraint** — shared by every mechanism, which all misfire on the same biased CLIP write stream.

## Reproducibility

- Code: `models/` adapters, `configs/<method>/` per-dataset configs, `experiments/slurm/` array scripts, `scripts/` analyzers and unit tests (all referenced inside the analysis docs).
- Results: `outputs_repr_upgrade/result_*.txt` (small, human-readable); per-sample records under `outputs_repr_upgrade/records_*` and `outputs_limitation_analysis_0faafb9/` (large, not for version control).
- Backbone: ViT-B/16 (CLIP Surgery); datasets: dtd, oxford_flowers, oxford_pets; seeds 1–4; batch size 1 by design.
