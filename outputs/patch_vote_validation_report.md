# Experiment J — Deep Validation of the Shortlisted Aggregation Signal(s) — Raw Results

Shortlist from Experiment I: `mean` (control), `topk20`, `pmean16`, `surgery_masked_mean`. All four checks below (multi-seed, margin-decile, per-class, cross-signal) ran on all three datasets; the qualitative spot-check pulled concrete examples for `topk20`.

## 1. Multi-seed replication

Standalone accuracy and agreement rate are **exactly identical across all 4 seeds** for every variant/dataset (confirmed empirically, not just theoretically) -- they depend only on each image's own zero-shot CLIP logits and patch embeddings, neither of which depends on test-set processing order. Only write purity (which depends on the specific final bank state, itself order-dependent) varies by seed -- and it stays in a tight, consistent band.

### dtd

| Variant | Standalone acc | Agree rate | Purity-agree range (s1-s4) | Purity-disagree range (s1-s4) | Gap range (s1-s4) |
|---|---|---|---|---|---|
| mean | 40.78% | 59.6% | 69.3-72.2% | 40.1-43.2% | +26.1 to +31.6pp |
| topk20 | 41.31% | 60.0% | 70.0-72.5% | 35.5-39.2% | +30.8 to +37.1pp |
| pmean16 | 41.31% | 61.0% | 68.7-71.2% | 38.6-42.6% | +26.0 to +32.3pp |
| surgery_masked_mean | 41.08% | 59.8% | 68.9-71.4% | 39.1-42.9% | +26.0 to +32.0pp |

### oxford_flowers

| Variant | Standalone acc | Agree rate | Purity-agree range (s1-s4) | Purity-disagree range (s1-s4) | Gap range (s1-s4) |
|---|---|---|---|---|---|
| mean | 20.71% | 20.5% | 95.8-96.1% | 67.2-69.0% | +26.9 to +28.9pp |
| topk20 | 28.99% | 31.6% | 85.4-85.8% | 67.3-69.6% | +16.1 to +18.4pp |
| pmean16 | 30.69% | 32.8% | 85.9-86.4% | 66.6-68.9% | +17.4 to +19.8pp |
| surgery_masked_mean | 26.55% | 28.7% | 85.9-86.7% | 67.9-69.8% | +16.6 to +18.4pp |

### oxford_pets

| Variant | Standalone acc | Agree rate | Purity-agree range (s1-s4) | Purity-disagree range (s1-s4) | Gap range (s1-s4) |
|---|---|---|---|---|---|
| mean | 30.20% | 30.0% | 93.1-94.2% | 81.4-82.0% | +11.6 to +12.2pp |
| topk20 | 65.41% | 64.8% | 93.2-94.4% | 70.2-72.3% | +20.9 to +23.4pp |
| pmean16 | 64.87% | 63.9% | 93.3-94.7% | 70.6-72.4% | +20.9 to +23.2pp |
| surgery_masked_mean | 60.53% | 59.9% | 93.0-94.4% | 72.9-74.5% | +18.5 to +20.5pp |

## 2. Margin-graded analysis (topk20, decile bins)

The variant's own top1-top2 margin (softmax over its pooled scores) carries real graded information, not just a binary agree/disagree split -- CLS accuracy and write purity trend upward with margin, most cleanly on dtd and oxford_pets.

### dtd

| Decile | Margin range | n | CLS acc | Write purity (n) |
|---|---|---|---|---|
| 0 | [0.000, 0.032] | 169 | 23.7% | 30.0% (n=20) |
| 1 | [0.033, 0.078] | 169 | 27.2% | 50.0% (n=26) |
| 2 | [0.078, 0.146] | 169 | 29.6% | 61.8% (n=55) |
| 3 | [0.147, 0.229] | 169 | 26.0% | 48.5% (n=66) |
| 4 | [0.230, 0.358] | 169 | 29.6% | 54.7% (n=95) |
| 5 | [0.358, 0.563] | 169 | 40.8% | 64.6% (n=127) |
| 6 | [0.565, 0.756] | 169 | 54.4% | 71.7% (n=184) |
| 7 | [0.756, 0.899] | 169 | 62.7% | 80.6% (n=217) |
| 8 | [0.900, 0.974] | 169 | 62.7% | 76.1% (n=230) |
| 9 | [0.974, 0.999] | 171 | 83.6% | 88.7% (n=239) |

### oxford_flowers

| Decile | Margin range | n | CLS acc | Write purity (n) |
|---|---|---|---|---|
| 0 | [0.000, 0.043] | 246 | 66.3% | 73.3% (n=75) |
| 1 | [0.043, 0.094] | 246 | 65.0% | 82.1% (n=84) |
| 2 | [0.094, 0.160] | 246 | 67.1% | 87.2% (n=109) |
| 3 | [0.160, 0.242] | 246 | 66.7% | 76.8% (n=151) |
| 4 | [0.243, 0.349] | 246 | 67.9% | 93.7% (n=127) |
| 5 | [0.349, 0.489] | 246 | 77.2% | 85.9% (n=149) |
| 6 | [0.490, 0.643] | 246 | 75.6% | 86.9% (n=122) |
| 7 | [0.643, 0.800] | 246 | 72.4% | 91.7% (n=157) |
| 8 | [0.800, 0.914] | 246 | 69.1% | 85.4% (n=151) |
| 9 | [0.915, 1.000] | 249 | 79.9% | 86.4% (n=330) |

### oxford_pets

| Decile | Margin range | n | CLS acc | Write purity (n) |
|---|---|---|---|---|
| 0 | [0.000, 0.073] | 366 | 83.6% | 80.5% (n=498) |
| 1 | [0.073, 0.150] | 366 | 82.5% | 84.2% (n=552) |
| 2 | [0.150, 0.244] | 366 | 83.6% | 89.3% (n=713) |
| 3 | [0.244, 0.366] | 366 | 85.0% | 93.7% (n=810) |
| 4 | [0.366, 0.493] | 366 | 87.7% | 93.3% (n=801) |
| 5 | [0.494, 0.640] | 366 | 89.6% | 95.2% (n=929) |
| 6 | [0.640, 0.771] | 366 | 91.8% | 94.1% (n=1089) |
| 7 | [0.772, 0.892] | 366 | 91.0% | 95.2% (n=1294) |
| 8 | [0.893, 0.962] | 366 | 95.6% | 96.8% (n=1280) |
| 9 | [0.962, 1.000] | 375 | 99.7% | 99.7% (n=1178) |

## 3. Per-class breakdown (topk20)

Agreement rate correlates with CLIP's own per-class baseline accuracy (moderate positive correlation -- the corroboration signal is somewhat concentrated in classes CLIP is already decent at) but far from perfectly, meaning it also captures real per-image variation within a class, not just a repackaging of 'this is a hard class.' Per-class numbers are noisy at n=13-100 samples/class, so individual reversed gaps (e.g. a class with a negative gap) are expected noise, not a sign the signal breaks down -- judge by the aggregate correlation and the class-pooled numbers in Experiment I, not single-class outliers.

### dtd (n_classes=47, corr(baseline_acc, agreement_rate) = 0.650)

| Classname | n | CLIP baseline acc | Agreement rate | Gap |
|---|---|---|---|---|
| lacelike | 36 | 0.0% | 55.6% | 0.0pp |
| flecked | 36 | 0.0% | 30.6% | 0.0pp |
| pitted | 36 | 0.0% | 47.2% | 0.0pp |
| lined | 36 | 0.0% | 77.8% | 0.0pp |
| bumpy | 36 | 2.8% | 41.7% | -4.8pp |
| ... |  |  |  |  |
| cobwebbed | 36 | 91.7% | 91.7% | 27.3pp |
| bubbly | 36 | 91.7% | 83.3% | 30.0pp |
| chequered | 36 | 100.0% | 100.0% | N/A |
| paisley | 36 | 100.0% | 100.0% | N/A |
| knitted | 36 | 100.0% | 100.0% | N/A |

### oxford_flowers (n_classes=102, corr(baseline_acc, agreement_rate) = 0.246)

| Classname | n | CLIP baseline acc | Agreement rate | Gap |
|---|---|---|---|---|
| geranium | 34 | 0.0% | 0.0% | N/A |
| great masterwort | 17 | 0.0% | 0.0% | N/A |
| colt's foot | 26 | 0.0% | 0.0% | N/A |
| globe-flower | 13 | 0.0% | 23.1% | 0.0pp |
| hard-leaved pocket orchid | 18 | 0.0% | 5.6% | 0.0pp |
| ... |  |  |  |  |
| garden phlox | 14 | 100.0% | 0.0% | N/A |
| king protea | 15 | 100.0% | 20.0% | 0.0pp |
| daffodil | 17 | 100.0% | 100.0% | N/A |
| osteospermum | 19 | 100.0% | 0.0% | N/A |
| grape hyacinth | 13 | 100.0% | 46.2% | 0.0pp |

### oxford_pets (n_classes=37, corr(baseline_acc, agreement_rate) = 0.455)

| Classname | n | CLIP baseline acc | Agreement rate | Gap |
|---|---|---|---|---|
| bombay | 88 | 27.3% | 11.4% | -19.5pp |
| birman | 100 | 46.0% | 42.0% | 60.3pp |
| american_pit_bull_terrier | 100 | 67.0% | 59.0% | -10.5pp |
| ragdoll | 100 | 72.0% | 13.0% | -65.1pp |
| persian | 100 | 72.0% | 28.0% | 14.1pp |
| ... |  |  |  |  |
| havanese | 100 | 99.0% | 68.0% | -1.5pp |
| german_shorthaired | 100 | 100.0% | 77.0% | 0.0pp |
| pomeranian | 100 | 100.0% | 99.0% | 0.0pp |
| shiba_inu | 100 | 100.0% | 97.0% | 0.0pp |
| samoyed | 100 | 100.0% | 100.0% | N/A |

## 4. Cross-signal correlation (topk20)

Two checks for whether agreement is genuinely new information or a repackaging of a signal this investigation already has:

**(a) vs. the confident-vs-ambiguous CLIP write-margin split** (clip_write_margin >= 0.2 = confident): agreement predicts purity **within both confidence buckets**, not just by proxy through confidence -- i.e. this is additive information, not a repackaging.

| Dataset | Confident+agree | Confident+disagree | Ambiguous+agree | Ambiguous+disagree |
|---|---|---|---|---|
| dtd | 75.2% | 36.9% | 50.7% | 33.5% |
| oxford_flowers | 90.7% | 77.1% | 48.0% | 41.3% |
| oxford_pets | 96.5% | 81.0% | 58.9% | 38.1% |

**(b) vs. Experiment F's per-prototype confusability margin** (intra-class match minus nearest-cross-class match, per touched prototype): essentially **no difference** between agree/disagree -- agreement is not just picking out geometrically-more-separable prototypes; it's a different axis of information entirely (about whether *this image's* patches look like the claimed class, not about how separable that class's bank is in general).

| Dataset | Mean confusability margin (agree) | Mean confusability margin (disagree) |
|---|---|---|
| dtd | -0.2838 | -0.2790 |
| oxford_flowers | -0.3079 | -0.3052 |
| oxford_pets | -0.3229 | -0.3280 |

## 5. Qualitative spot-check (topk20, oxford_pets)

128 rescue cases, 995 corrupt cases total. Sample:

Rescue (patch-vote right, CLIP wrong):
- target=`bengal`, CLIP said `abyssinian`, patch-vote said `bengal`
- target=`birman`, CLIP said `ragdoll`, patch-vote said `birman`
- target=`beagle`, CLIP said `basset_hound`, patch-vote said `beagle`
- target=`birman`, CLIP said `ragdoll`, patch-vote said `birman`
- target=`birman`, CLIP said `ragdoll`, patch-vote said `birman`

Corrupt (CLIP right, patch-vote wrong):
- target=`wheaten_terrier`, CLIP said `wheaten_terrier`, patch-vote said `pug`
- target=`japanese_chin`, CLIP said `japanese_chin`, patch-vote said `pomeranian`
- target=`ragdoll`, CLIP said `ragdoll`, patch-vote said `siamese`
- target=`english_setter`, CLIP said `english_setter`, patch-vote said `great_pyrenees`
- target=`american_pit_bull_terrier`, CLIP said `american_pit_bull_terrier`, patch-vote said `staffordshire_bull_terrier`

Both sets are dominated by visually-similar breed confusions (birman vs. ragdoll, bengal vs. abyssinian, pit-bull vs. staffordshire) -- consistent with `topk20` genuinely doing fine-grained visual discrimination, just not reliably enough to trust alone.

# Experiment J — Deep Validation of the Shortlisted Aggregation Signal(s) — Raw Results

Shortlist from Experiment I: `mean` (control), `topk20`, `pmean16`, `surgery_masked_mean`, plus `otsu_mean` (added after the first J pass, at the user's request, as a hyperparameter-free alternative to `topk20`'s fixed k). All checks below ran for all five variants; tables display `topk20` and `otsu_mean` since those are the two candidates carried into Experiment K.

## 1. Multi-seed replication

### dtd

| Variant | Standalone acc | Agree rate | Purity-agree range (s1-s4) | Purity-disagree range (s1-s4) | Gap range (s1-s4) |
|---|---|---|---|---|---|
| topk20 | 41.31% | 60.0% | 70.0-72.5% | 35.5-39.2% | +30.8 to +37.1pp |
| otsu_mean | 41.61% | 60.5% | 69.6-71.9% | 36.4-40.3% | +29.3 to +35.5pp |

### oxford_flowers

| Variant | Standalone acc | Agree rate | Purity-agree range (s1-s4) | Purity-disagree range (s1-s4) | Gap range (s1-s4) |
|---|---|---|---|---|---|
| topk20 | 28.99% | 31.6% | 85.4-85.8% | 67.3-69.6% | +16.1 to +18.4pp |
| otsu_mean | 29.11% | 30.8% | 86.1-86.9% | 67.2-69.0% | +17.9 to +19.0pp |

### oxford_pets

| Variant | Standalone acc | Agree rate | Purity-agree range (s1-s4) | Purity-disagree range (s1-s4) | Gap range (s1-s4) |
|---|---|---|---|---|---|
| topk20 | 65.41% | 64.8% | 93.2-94.4% | 70.2-72.3% | +20.9 to +23.4pp |
| otsu_mean | 61.11% | 60.1% | 93.3-94.6% | 72.5-74.1% | +19.3 to +21.1pp |

## 2. Margin-graded analysis, per-class correlation, cross-signal (otsu_mean)

Same pattern as `topk20` on every check:

| Dataset | Margin-decile trend | Per-class corr(baseline_acc, agree_rate) | Confident+agree/disagree purity | Ambiguous+agree/disagree purity | Confusability margin agree/disagree |
|---|---|---|---|---|---|
| dtd | 22%→83% | 0.662 | 74.6% / 38.6% | 50.7% / 33.5% | -0.285 / -0.274 |
| oxford_flowers | 62%→82% | 0.240 | 90.9% / 76.9% | 53.2% / 40.4% | -0.305 / -0.307 |
| oxford_pets | 82%→100% | 0.374 | 96.7% / 82.5% | 58.5% / 38.2% | -0.325 / -0.325 |

## Bottom line: `otsu_mean` passes every check `topk20` passed

Multi-seed stable, margin carries graded (mostly monotonic) information, moderate per-class correlation without reducing to it, and confirmed non-redundant with both the write-confidence split and Experiment F's confusability margin -- same conclusions, same strength, as `topk20`. Practical difference from Experiment I still holds: `otsu_mean` is stronger/tied on dtd and oxford_flowers, `topk20`/`pmean16` are stronger on oxford_pets. Both carried into Experiment K as parallel candidates, per the user's request, rather than picking one now.

