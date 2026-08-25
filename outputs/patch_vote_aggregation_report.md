# Experiment I — Patch-Level Aggregation Signal Comparison — Raw Results

13 pooling variants, all computed from the same single CLIP Surgery forward pass per image (patch embeddings + one `compute_surgery_scores()` call, no augmentation, no second model). `mean` reproduces Experiment H exactly (verified). `otsu_mean` added after the first pass, at the user's request, as a hyperparameter-free alternative to `topk20` -- Otsu's method finds the threshold that best splits each class's patch similarities into a high/low group, with no k or z-score cutoff to pick. Deltas/gaps below are `cls_acc_when_agree - cls_acc_when_disagree`.

## dtd (n=1692)

| Variant | Standalone acc | Agree rate | CLS acc (agree/disagree) | Gap | Write purity (agree/disagree) | Rescue rate | Corrupt rate |
|---|---|---|---|---|---|---|---|
| mean | 40.78% | 59.6% | 58.43% / 22.95% | +35.48pp | 72.19% / 40.85% | 10.68% | 21.05% |
| max | 39.83% | 55.9% | 61.59% / 21.95% | +39.63pp | 72.47% / 38.78% | 9.73% | 21.98% |
| topk3 | 40.90% | 58.1% | 60.63% / 21.16% | +39.47pp | 72.68% / 36.01% | 10.15% | 20.11% |
| topk5 | 41.08% | 58.7% | 60.12% / 21.32% | +38.80pp | 72.68% / 35.60% | 10.36% | 19.97% |
| topk10 | 41.08% | 59.6% | 59.62% / 21.20% | +38.42pp | 72.18% / 35.62% | 9.94% | 19.44% |
| topk20 | 41.31% | 60.0% | 59.31% / 21.27% | +38.04pp | 72.52% / 35.47% | 10.25% | 19.30% |
| pmean2 | 41.08% | 59.8% | 58.65% / 22.47% | +36.19pp | 72.38% / 39.90% | 10.78% | 20.51% |
| pmean4 | 41.25% | 60.4% | 58.61% / 21.94% | +36.67pp | 72.33% / 38.19% | 10.47% | 19.71% |
| pmean8 | 41.49% | 61.0% | 58.53% / 21.52% | +37.01pp | 72.09% / 37.37% | 10.36% | 19.03% |
| pmean16 | 41.31% | 61.0% | 58.33% / 21.82% | +36.52pp | 71.24% / 38.90% | 10.25% | 19.30% |
| surgery_masked_mean | 41.08% | 59.8% | 59.15% / 21.73% | +37.42pp | 71.38% / 39.36% | 10.25% | 19.84% |
| surgery_score_mean | 40.78% | 59.6% | 58.43% / 22.95% | +35.48pp | 72.19% / 40.85% | 10.68% | 21.05% |
| otsu_mean | 41.61% | 60.5% | 59.24% / 20.93% | +38.31pp | 71.95% / 36.44% | 10.36% | 18.77% |

## oxford_flowers (n=2463)

| Variant | Standalone acc | Agree rate | CLS acc (agree/disagree) | Gap | Write purity (agree/disagree) | Rescue rate | Corrupt rate |
|---|---|---|---|---|---|---|---|
| mean | 20.71% | 20.5% | 94.66% / 64.54% | +30.13pp | 95.91% / 69.00% | 4.30% | 72.50% |
| max | 23.67% | 25.7% | 84.86% / 65.83% | +19.03pp | 84.72% / 71.02% | 6.24% | 69.12% |
| topk3 | 25.42% | 27.5% | 85.38% / 65.17% | +20.20pp | 85.44% / 70.52% | 6.66% | 66.82% |
| topk5 | 26.15% | 28.4% | 85.14% / 65.00% | +20.14pp | 85.10% / 70.43% | 6.66% | 65.79% |
| topk10 | 27.16% | 29.5% | 85.26% / 64.65% | +20.61pp | 85.46% / 70.10% | 6.93% | 64.47% |
| topk20 | 28.99% | 31.6% | 85.62% / 63.84% | +21.79pp | 85.64% / 69.58% | 6.52% | 61.71% |
| pmean2 | 22.61% | 22.5% | 93.85% / 64.03% | +29.82pp | 94.97% / 68.62% | 5.27% | 70.21% |
| pmean4 | 25.66% | 26.3% | 91.19% / 63.44% | +27.75pp | 92.07% / 68.37% | 5.83% | 66.13% |
| pmean8 | 29.44% | 30.9% | 87.80% / 63.08% | +24.71pp | 88.15% / 68.47% | 7.77% | 61.60% |
| pmean16 | 30.69% | 32.8% | 86.51% / 63.02% | +23.49pp | 86.28% / 68.86% | 7.91% | 59.87% |
| surgery_masked_mean | 26.55% | 28.7% | 86.12% / 64.54% | +21.58pp | 86.38% / 69.79% | 6.38% | 65.10% |
| surgery_score_mean | 20.71% | 20.5% | 94.66% / 64.54% | +30.13pp | 95.91% / 69.00% | 4.30% | 72.50% |
| otsu_mean | 29.11% | 30.8% | 86.54% / 63.70% | +22.85pp | 86.94% / 69.03% | 8.46% | 62.34% |

## oxford_pets (n=3669)

| Variant | Standalone acc | Agree rate | CLS acc (agree/disagree) | Gap | Write purity (agree/disagree) | Rescue rate | Corrupt rate |
|---|---|---|---|---|---|---|---|
| mean | 30.20% | 30.0% | 95.28% / 86.37% | +8.91pp | 93.36% / 81.40% | 14.68% | 67.89% |
| max | 62.55% | 62.2% | 95.53% / 78.35% | +17.18pp | 93.50% / 72.09% | 28.36% | 33.24% |
| topk3 | 63.75% | 63.3% | 95.61% / 77.73% | +17.88pp | 93.58% / 71.22% | 29.60% | 32.05% |
| topk5 | 64.16% | 63.8% | 95.35% / 77.92% | +17.43pp | 93.31% / 71.40% | 30.10% | 31.65% |
| topk10 | 65.03% | 64.5% | 95.52% / 77.30% | +18.22pp | 93.59% / 70.53% | 31.59% | 30.85% |
| topk20 | 65.41% | 64.8% | 95.50% / 77.13% | +18.37pp | 93.65% / 70.22% | 31.84% | 30.46% |
| pmean2 | 36.09% | 35.9% | 95.07% / 85.67% | +9.40pp | 93.24% / 80.40% | 17.66% | 61.65% |
| pmean4 | 47.26% | 46.7% | 95.16% / 83.67% | +11.49pp | 93.26% / 77.89% | 25.37% | 50.05% |
| pmean8 | 60.83% | 60.3% | 95.35% / 79.46% | +15.88pp | 93.38% / 72.88% | 30.35% | 35.41% |
| pmean16 | 64.87% | 63.9% | 95.69% / 77.30% | +18.39pp | 93.76% / 70.59% | 34.33% | 31.37% |
| surgery_masked_mean | 60.53% | 59.9% | 95.40% / 79.57% | +15.84pp | 93.43% / 72.93% | 31.34% | 35.87% |
| surgery_score_mean | 30.20% | 30.0% | 95.28% / 86.37% | +8.91pp | 93.36% / 81.40% | 14.68% | 67.89% |
| otsu_mean | 61.11% | 60.1% | 95.60% / 79.18% | +16.42pp | 93.62% / 72.51% | 33.58% | 35.51% |

## Note: `surgery_score_mean` is mathematically degenerate

`surgery_score_mean` is bit-for-bit identical to `mean` on every metric in every dataset above. This isn't a coincidence or a bug: `compute_surgery_scores(..., filter_mode="surgery_no_labels")` calls `clip_feature_surgery(image_features, text_features, redundant_feats=empty_text_feat)`, which for the `redundant_feats` branch is just `similarity = image_features @ (text_features - redundant_feats).T` -- a *linear* dot product against a text embedding shifted by the same class-independent constant for every class. Mean-pooling that over patches gives `mean_pool(sims) - K` for a constant `K` that doesn't depend on class -- same argmax, same softmax (shift-invariant), same everything as plain `mean`. `surgery_masked_mean` does NOT have this problem (it uses the same scores as a per-class *mask* over patches, not as the score itself, and the per-patch constant does change which patches get selected per class), which is why it shows genuinely different numbers above.

## `otsu_mean` vs. `topk20`

`otsu_mean` **wins outright on dtd** (best standalone accuracy, agreement rate, gap, and corrupt rate of all 13 variants), is essentially tied with the leaders on oxford_flowers (best write purity and rescue rate of any variant, standalone accuracy within 0.1pp of `otsu_mean` — reads not much different from `topk20`), but trails `topk20`/`pmean16` on oxford_pets (61.1% vs 65.4% standalone accuracy). A quick check of how many patches Otsu actually keeps confirms it's genuinely adaptive, not a fixed count in disguise: median ~93-122 patches (out of 196) on dtd's whole-image textures vs. a much wider, more selective range (11-149) on oxford_pets, varying per image and per class.

