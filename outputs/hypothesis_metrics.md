# Hypothesis Metrics — stream-time accuracy vs patch-prototype growth

Two versions documented here, clearly labeled:

- **PART A (old code)**: generated from `outputs/records` (prior plan, BUGGY config path: single-gate, alpha-capped patch term). Full-DTD tables below are OLD-CODE.
- **PART B (follow-up revalidation, fixed code, deterministic loader, seed 1, records_v2)**: generated from `outputs/records_v2/subset` (Task 5, plan `patch-proto-revalidation`). Subset tables only. The point-5 bar (rho/delta) was NOT testable: Wave 3 full-DTD per-seed runs were skipped after the Wave 2 subset gate failed.

---

# PART A: Prior plan, old code (preserved history)

Hypothesis (pre-registered): *as you build better patch-level prototypes over time, classification performance increases.*

Decision rule (PatchModPTA-CS full-DTD records only, ablations excluded):

- SUPPORT if `delta_mean > +1.0 pp` **AND** `rho_mean > 0.2`
- REFUTE if `delta_mean < 0`
- INCONCLUSIVE otherwise

`acc_first` = mean(correct) over first 50% of stream; `acc_last` = mean(correct) over last 50%; `delta = acc_last - acc_first` (pp).
`rho` = Spearman(cluster-growth vs last-half per-class accuracy), masked to classes with nonzero clusters at stream end.

## Per-run stream metrics

| label | method | seed | n | acc_first (%) | acc_last (%) | delta (pp) | rho |
|---|---|---|---|---|---|---|---|
| PTA-CS-s1 | PTA | 1 | 1692 | 44.68 | 49.29 | 4.61 | 0.00 |
| PTA-CS-s2 | PTA | 2 | 1692 | 44.33 | 48.46 | 4.14 | 0.00 |
| PatchModPTA-CS-s1 | PatchModPTA | 1 | 1692 | 45.63 | 51.06 | 5.44 | -0.51 |
| PatchModPTA-CS-s2 | PatchModPTA | 2 | 1692 | 45.39 | 50.00 | 4.61 | -0.37 |
| ZeroShot-CS-s1 | ZeroShot | 1 | 1692 | 43.38 | 45.39 | 2.01 | 0.00 |
| ZeroShot-CS-s2 | ZeroShot | 2 | 1692 | 43.26 | 45.51 | 2.25 | 0.00 |

## Rolling-window accuracy

Window size = 100 samples (non-overlapping).

### PTA-CS-s1

| samples | n | acc (%) |
|---|---|---|
| [0:100) | 100 | 43.00 |
| [100:200) | 100 | 38.00 |
| [200:300) | 100 | 52.00 |
| [300:400) | 100 | 53.00 |
| [400:500) | 100 | 36.00 |
| [500:600) | 100 | 40.00 |
| [600:700) | 100 | 44.00 |
| [700:800) | 100 | 44.00 |
| [800:900) | 100 | 51.00 |
| [900:1000) | 100 | 59.00 |
| [1000:1100) | 100 | 37.00 |
| [1100:1200) | 100 | 47.00 |
| [1200:1300) | 100 | 48.00 |
| [1300:1400) | 100 | 50.00 |
| [1400:1500) | 100 | 47.00 |
| [1500:1600) | 100 | 57.00 |
| [1600:1692) | 92 | 53.26 |

### PTA-CS-s2

| samples | n | acc (%) |
|---|---|---|
| [0:100) | 100 | 40.00 |
| [100:200) | 100 | 46.00 |
| [200:300) | 100 | 45.00 |
| [300:400) | 100 | 38.00 |
| [400:500) | 100 | 43.00 |
| [500:600) | 100 | 48.00 |
| [600:700) | 100 | 53.00 |
| [700:800) | 100 | 42.00 |
| [800:900) | 100 | 43.00 |
| [900:1000) | 100 | 51.00 |
| [1000:1100) | 100 | 49.00 |
| [1100:1200) | 100 | 41.00 |
| [1200:1300) | 100 | 60.00 |
| [1300:1400) | 100 | 45.00 |
| [1400:1500) | 100 | 38.00 |
| [1500:1600) | 100 | 50.00 |
| [1600:1692) | 92 | 57.61 |

### PatchModPTA-CS-s1

| samples | n | acc (%) |
|---|---|---|
| [0:100) | 100 | 45.00 |
| [100:200) | 100 | 37.00 |
| [200:300) | 100 | 54.00 |
| [300:400) | 100 | 55.00 |
| [400:500) | 100 | 36.00 |
| [500:600) | 100 | 43.00 |
| [600:700) | 100 | 44.00 |
| [700:800) | 100 | 46.00 |
| [800:900) | 100 | 49.00 |
| [900:1000) | 100 | 62.00 |
| [1000:1100) | 100 | 39.00 |
| [1100:1200) | 100 | 50.00 |
| [1200:1300) | 100 | 50.00 |
| [1300:1400) | 100 | 54.00 |
| [1400:1500) | 100 | 46.00 |
| [1500:1600) | 100 | 58.00 |
| [1600:1692) | 92 | 54.35 |

### PatchModPTA-CS-s2

| samples | n | acc (%) |
|---|---|---|
| [0:100) | 100 | 43.00 |
| [100:200) | 100 | 52.00 |
| [200:300) | 100 | 47.00 |
| [300:400) | 100 | 39.00 |
| [400:500) | 100 | 41.00 |
| [500:600) | 100 | 49.00 |
| [600:700) | 100 | 53.00 |
| [700:800) | 100 | 41.00 |
| [800:900) | 100 | 44.00 |
| [900:1000) | 100 | 51.00 |
| [1000:1100) | 100 | 48.00 |
| [1100:1200) | 100 | 43.00 |
| [1200:1300) | 100 | 60.00 |
| [1300:1400) | 100 | 50.00 |
| [1400:1500) | 100 | 45.00 |
| [1500:1600) | 100 | 49.00 |
| [1600:1692) | 92 | 56.52 |

### ZeroShot-CS-s1

| samples | n | acc (%) |
|---|---|---|
| [0:100) | 100 | 44.00 |
| [100:200) | 100 | 37.00 |
| [200:300) | 100 | 51.00 |
| [300:400) | 100 | 50.00 |
| [400:500) | 100 | 32.00 |
| [500:600) | 100 | 41.00 |
| [600:700) | 100 | 43.00 |
| [700:800) | 100 | 46.00 |
| [800:900) | 100 | 44.00 |
| [900:1000) | 100 | 55.00 |
| [1000:1100) | 100 | 38.00 |
| [1100:1200) | 100 | 39.00 |
| [1200:1300) | 100 | 45.00 |
| [1300:1400) | 100 | 46.00 |
| [1400:1500) | 100 | 40.00 |
| [1500:1600) | 100 | 54.00 |
| [1600:1692) | 92 | 50.00 |

### ZeroShot-CS-s2

| samples | n | acc (%) |
|---|---|---|
| [0:100) | 100 | 42.00 |
| [100:200) | 100 | 54.00 |
| [200:300) | 100 | 42.00 |
| [300:400) | 100 | 36.00 |
| [400:500) | 100 | 40.00 |
| [500:600) | 100 | 42.00 |
| [600:700) | 100 | 52.00 |
| [700:800) | 100 | 39.00 |
| [800:900) | 100 | 45.00 |
| [900:1000) | 100 | 47.00 |
| [1000:1100) | 100 | 43.00 |
| [1100:1200) | 100 | 36.00 |
| [1200:1300) | 100 | 56.00 |
| [1300:1400) | 100 | 44.00 |
| [1400:1500) | 100 | 38.00 |
| [1500:1600) | 100 | 49.00 |
| [1600:1692) | 92 | 50.00 |

## Per-class cluster-growth vs last-half accuracy (Spearman rho)

### PTA-CS-s1  (rho = 0.00)


### PTA-CS-s2  (rho = 0.00)


### PatchModPTA-CS-s1  (rho = -0.51)

| class | n_clusters_end | last-half acc (%) |
|---|---|---|
| banded | 6 | 0.00 |
| blotchy | 3 | 7.69 |
| braided | 2 | 40.91 |
| bubbly | 2 | 100.00 |
| bumpy | 6 | 0.00 |
| chequered | 2 | 100.00 |
| cobwebbed | 4 | 95.65 |
| cracked | 5 | 75.00 |
| crosshatched | 8 | 40.00 |
| crystalline | 5 | 88.24 |
| dotted | 3 | 23.53 |
| fibrous | 6 | 52.38 |
| freckled | 3 | 81.82 |
| frilly | 2 | 60.00 |
| gauzy | 5 | 5.56 |
| grid | 4 | 50.00 |
| honeycombed | 2 | 68.42 |
| knitted | 2 | 100.00 |
| lacelike | 5 | 0.00 |
| marbled | 3 | 100.00 |
| matted | 5 | 25.00 |
| meshed | 2 | 35.29 |
| paisley | 2 | 100.00 |
| perforated | 7 | 76.19 |
| pleated | 3 | 52.94 |
| polka-dotted | 2 | 84.21 |
| porous | 3 | 64.71 |
| scaly | 2 | 87.50 |
| smeared | 3 | 50.00 |
| spiralled | 6 | 40.91 |
| sprinkled | 8 | 52.63 |
| stained | 5 | 33.33 |
| striped | 2 | 77.78 |
| studded | 2 | 85.00 |
| swirly | 3 | 66.67 |
| veined | 3 | 56.25 |
| waffled | 2 | 70.00 |
| woven | 4 | 47.06 |
| wrinkled | 3 | 50.00 |

*Masked (zero clusters at end): flecked, grooved, interlaced, lined, pitted, potholed, stratified, zigzagged*

### PatchModPTA-CS-s2  (rho = -0.37)

| class | n_clusters_end | last-half acc (%) |
|---|---|---|
| banded | 6 | 5.00 |
| blotchy | 3 | 4.76 |
| braided | 3 | 33.33 |
| bubbly | 4 | 89.47 |
| bumpy | 5 | 7.14 |
| chequered | 3 | 100.00 |
| cobwebbed | 3 | 94.12 |
| cracked | 5 | 81.82 |
| crosshatched | 8 | 35.29 |
| crystalline | 3 | 100.00 |
| dotted | 4 | 13.33 |
| fibrous | 4 | 50.00 |
| freckled | 4 | 66.67 |
| frilly | 2 | 56.25 |
| gauzy | 5 | 21.05 |
| grid | 4 | 53.33 |
| honeycombed | 3 | 50.00 |
| knitted | 2 | 100.00 |
| lacelike | 4 | 0.00 |
| marbled | 2 | 93.75 |
| matted | 5 | 22.22 |
| meshed | 2 | 42.86 |
| paisley | 4 | 100.00 |
| perforated | 7 | 76.19 |
| pleated | 3 | 70.59 |
| polka-dotted | 2 | 82.35 |
| porous | 2 | 78.95 |
| scaly | 3 | 89.47 |
| smeared | 3 | 66.67 |
| spiralled | 2 | 29.41 |
| sprinkled | 7 | 42.86 |
| stained | 6 | 18.18 |
| striped | 2 | 63.16 |
| studded | 3 | 73.68 |
| swirly | 4 | 59.09 |
| veined | 2 | 57.14 |
| waffled | 3 | 70.59 |
| woven | 3 | 66.67 |
| wrinkled | 3 | 50.00 |

*Masked (zero clusters at end): flecked, grooved, interlaced, lined, pitted, potholed, stratified, zigzagged*

### ZeroShot-CS-s1  (rho = 0.00)


### ZeroShot-CS-s2  (rho = 0.00)


## Flip diagnostics (offline, diagnostic only — NOT part of the decision rule)

- `patch_flip`: argmax(final) != argmax(clip)
- `flip_vs_img`: argmax(final) != argmax(clip + tau_img*image_proto) (— when image_proto / tau unavailable)
- `patch_only_flip`: argmax(patch_proto) != argmax(clip), masked when the class has zero prototypes

### PTA-CS-s1  (tau_img = 100.0)

| class | n | patch_flip (%) | flip_vs_img (%) | patch_only_flip (%) |
|---|---|---|---|---|
| banded | 36 | 5.56 | 5.56 | — |
| blotchy | 36 | 44.44 | 41.67 | — |
| braided | 36 | 16.67 | 16.67 | — |
| bubbly | 36 | 5.56 | 5.56 | — |
| bumpy | 36 | 36.11 | 36.11 | — |
| chequered | 36 | 2.78 | 2.78 | — |
| cobwebbed | 36 | 0.00 | 0.00 | — |
| cracked | 36 | 13.89 | 13.89 | — |
| crosshatched | 36 | 33.33 | 33.33 | — |
| crystalline | 36 | 8.33 | 8.33 | — |
| dotted | 36 | 19.44 | 19.44 | — |
| fibrous | 36 | 52.78 | 52.78 | — |
| flecked | 36 | 38.89 | 38.89 | — |
| freckled | 36 | 27.78 | 27.78 | — |
| frilly | 36 | 16.67 | 16.67 | — |
| gauzy | 36 | 30.56 | 30.56 | — |
| grid | 36 | 22.22 | 22.22 | — |
| grooved | 36 | 52.78 | 52.78 | — |
| honeycombed | 36 | 27.78 | 27.78 | — |
| interlaced | 36 | 25.00 | 25.00 | — |
| knitted | 36 | 0.00 | 0.00 | — |
| lacelike | 36 | 13.89 | 16.67 | — |
| lined | 36 | 16.67 | 16.67 | — |
| marbled | 36 | 13.89 | 13.89 | — |
| matted | 36 | 38.89 | 38.89 | — |
| meshed | 36 | 27.78 | 27.78 | — |
| paisley | 36 | 0.00 | 0.00 | — |
| perforated | 36 | 11.11 | 11.11 | — |
| pitted | 36 | 41.67 | 41.67 | — |
| pleated | 36 | 22.22 | 22.22 | — |
| polka-dotted | 36 | 5.56 | 5.56 | — |
| porous | 36 | 25.00 | 25.00 | — |
| potholed | 36 | 61.11 | 61.11 | — |
| scaly | 36 | 19.44 | 19.44 | — |
| smeared | 36 | 41.67 | 41.67 | — |
| spiralled | 36 | 25.00 | 25.00 | — |
| sprinkled | 36 | 25.00 | 25.00 | — |
| stained | 36 | 30.56 | 30.56 | — |
| stratified | 36 | 63.89 | 63.89 | — |
| striped | 36 | 30.56 | 30.56 | — |
| studded | 36 | 11.11 | 11.11 | — |
| swirly | 36 | 22.22 | 22.22 | — |
| veined | 36 | 25.00 | 25.00 | — |
| waffled | 36 | 8.33 | 8.33 | — |
| woven | 36 | 30.56 | 27.78 | — |
| wrinkled | 36 | 44.44 | 44.44 | — |
| zigzagged | 36 | 55.56 | 55.56 | — |

### PTA-CS-s2  (tau_img = 100.0)

| class | n | patch_flip (%) | flip_vs_img (%) | patch_only_flip (%) |
|---|---|---|---|---|
| banded | 36 | 8.33 | 8.33 | — |
| blotchy | 36 | 44.44 | 44.44 | — |
| braided | 36 | 13.89 | 13.89 | — |
| bubbly | 36 | 2.78 | 2.78 | — |
| bumpy | 36 | 36.11 | 36.11 | — |
| chequered | 36 | 0.00 | 0.00 | — |
| cobwebbed | 36 | 0.00 | 0.00 | — |
| cracked | 36 | 13.89 | 13.89 | — |
| crosshatched | 36 | 30.56 | 30.56 | — |
| crystalline | 36 | 8.33 | 8.33 | — |
| dotted | 36 | 22.22 | 22.22 | — |
| fibrous | 36 | 38.89 | 38.89 | — |
| flecked | 36 | 30.56 | 30.56 | — |
| freckled | 36 | 16.67 | 16.67 | — |
| frilly | 36 | 11.11 | 11.11 | — |
| gauzy | 36 | 27.78 | 27.78 | — |
| grid | 36 | 25.00 | 25.00 | — |
| grooved | 36 | 41.67 | 41.67 | — |
| honeycombed | 36 | 22.22 | 22.22 | — |
| interlaced | 36 | 30.56 | 30.56 | — |
| knitted | 36 | 0.00 | 0.00 | — |
| lacelike | 36 | 16.67 | 16.67 | — |
| lined | 36 | 22.22 | 22.22 | — |
| marbled | 36 | 13.89 | 13.89 | — |
| matted | 36 | 41.67 | 41.67 | — |
| meshed | 36 | 36.11 | 36.11 | — |
| paisley | 36 | 0.00 | 0.00 | — |
| perforated | 36 | 5.56 | 5.56 | — |
| pitted | 36 | 44.44 | 44.44 | — |
| pleated | 36 | 25.00 | 25.00 | — |
| polka-dotted | 36 | 5.56 | 5.56 | — |
| porous | 36 | 19.44 | 19.44 | — |
| potholed | 36 | 47.22 | 47.22 | — |
| scaly | 36 | 16.67 | 16.67 | — |
| smeared | 36 | 58.33 | 58.33 | — |
| spiralled | 36 | 27.78 | 27.78 | — |
| sprinkled | 36 | 19.44 | 19.44 | — |
| stained | 36 | 41.67 | 41.67 | — |
| stratified | 36 | 36.11 | 36.11 | — |
| striped | 36 | 25.00 | 25.00 | — |
| studded | 36 | 13.89 | 13.89 | — |
| swirly | 36 | 27.78 | 27.78 | — |
| veined | 36 | 25.00 | 25.00 | — |
| waffled | 36 | 2.78 | 2.78 | — |
| woven | 36 | 33.33 | 33.33 | — |
| wrinkled | 36 | 44.44 | 44.44 | — |
| zigzagged | 36 | 50.00 | 50.00 | — |

### PatchModPTA-CS-s1  (tau_img = 80.0)

| class | n | patch_flip (%) | flip_vs_img (%) | patch_only_flip (%) |
|---|---|---|---|---|
| banded | 36 | 2.78 | 5.56 | 20.00 |
| blotchy | 36 | 38.89 | 33.33 | 72.73 |
| braided | 36 | 13.89 | 13.89 | 45.71 |
| bubbly | 36 | 5.56 | 5.56 | 11.43 |
| bumpy | 36 | 27.78 | 25.00 | 78.26 |
| chequered | 36 | 0.00 | 2.78 | 11.43 |
| cobwebbed | 36 | 2.78 | 0.00 | 8.57 |
| cracked | 36 | 13.89 | 11.11 | 57.14 |
| crosshatched | 36 | 36.11 | 30.56 | 70.37 |
| crystalline | 36 | 8.33 | 8.33 | 11.43 |
| dotted | 36 | 13.89 | 19.44 | 29.03 |
| fibrous | 36 | 55.56 | 52.78 | 86.36 |
| flecked | 36 | 38.89 | 38.89 | — |
| freckled | 36 | 30.56 | 25.00 | 34.29 |
| frilly | 36 | 13.89 | 19.44 | 51.43 |
| gauzy | 36 | 25.00 | 27.78 | 59.38 |
| grid | 36 | 27.78 | 25.00 | 55.88 |
| grooved | 36 | 50.00 | 47.22 | — |
| honeycombed | 36 | 25.00 | 22.22 | 45.71 |
| interlaced | 36 | 25.00 | 25.00 | — |
| knitted | 36 | 0.00 | 0.00 | 0.00 |
| lacelike | 36 | 16.67 | 11.11 | 55.17 |
| lined | 36 | 11.11 | 16.67 | — |
| marbled | 36 | 13.89 | 13.89 | 22.86 |
| matted | 36 | 36.11 | 33.33 | 87.10 |
| meshed | 36 | 36.11 | 27.78 | 61.54 |
| paisley | 36 | 0.00 | 0.00 | 8.57 |
| perforated | 36 | 16.67 | 11.11 | 40.00 |
| pitted | 36 | 38.89 | 36.11 | — |
| pleated | 36 | 22.22 | 22.22 | 60.00 |
| polka-dotted | 36 | 8.33 | 2.78 | 19.44 |
| porous | 36 | 25.00 | 25.00 | 68.57 |
| potholed | 36 | 50.00 | 55.56 | — |
| scaly | 36 | 22.22 | 13.89 | 42.86 |
| smeared | 36 | 52.78 | 38.89 | 70.59 |
| spiralled | 36 | 16.67 | 22.22 | 42.86 |
| sprinkled | 36 | 25.00 | 19.44 | 43.75 |
| stained | 36 | 25.00 | 27.78 | 74.29 |
| stratified | 36 | 50.00 | 55.56 | — |
| striped | 36 | 19.44 | 22.22 | 52.78 |
| studded | 36 | 11.11 | 8.33 | 22.86 |
| swirly | 36 | 27.78 | 16.67 | 41.67 |
| veined | 36 | 22.22 | 27.78 | 54.29 |
| waffled | 36 | 8.33 | 5.56 | 14.29 |
| woven | 36 | 33.33 | 27.78 | 62.86 |
| wrinkled | 36 | 41.67 | 44.44 | 48.48 |
| zigzagged | 36 | 47.22 | 50.00 | — |

### PatchModPTA-CS-s2  (tau_img = 80.0)

| class | n | patch_flip (%) | flip_vs_img (%) | patch_only_flip (%) |
|---|---|---|---|---|
| banded | 36 | 8.33 | 8.33 | 15.15 |
| blotchy | 36 | 44.44 | 36.11 | 66.67 |
| braided | 36 | 19.44 | 16.67 | 40.62 |
| bubbly | 36 | 5.56 | 0.00 | 14.29 |
| bumpy | 36 | 30.56 | 33.33 | 72.73 |
| chequered | 36 | 0.00 | 0.00 | 14.29 |
| cobwebbed | 36 | 2.78 | 0.00 | 8.57 |
| cracked | 36 | 13.89 | 8.33 | 54.29 |
| crosshatched | 36 | 41.67 | 33.33 | 73.53 |
| crystalline | 36 | 8.33 | 8.33 | 13.89 |
| dotted | 36 | 19.44 | 22.22 | 23.53 |
| fibrous | 36 | 41.67 | 38.89 | 85.71 |
| flecked | 36 | 36.11 | 30.56 | — |
| freckled | 36 | 16.67 | 16.67 | 25.71 |
| frilly | 36 | 13.89 | 8.33 | 48.57 |
| gauzy | 36 | 36.11 | 27.78 | 69.57 |
| grid | 36 | 36.11 | 25.00 | 57.14 |
| grooved | 36 | 50.00 | 38.89 | — |
| honeycombed | 36 | 16.67 | 22.22 | 42.86 |
| interlaced | 36 | 30.56 | 30.56 | — |
| knitted | 36 | 2.78 | 0.00 | 8.57 |
| lacelike | 36 | 13.89 | 8.33 | 38.89 |
| lined | 36 | 13.89 | 22.22 | — |
| marbled | 36 | 13.89 | 11.11 | 36.36 |
| matted | 36 | 36.11 | 27.78 | 92.59 |
| meshed | 36 | 44.44 | 36.11 | 80.00 |
| paisley | 36 | 0.00 | 0.00 | 25.71 |
| perforated | 36 | 19.44 | 5.56 | 68.57 |
| pitted | 36 | 44.44 | 38.89 | — |
| pleated | 36 | 27.78 | 25.00 | 55.88 |
| polka-dotted | 36 | 5.56 | 5.56 | 14.29 |
| porous | 36 | 25.00 | 16.67 | 51.52 |
| potholed | 36 | 47.22 | 50.00 | — |
| scaly | 36 | 16.67 | 19.44 | 42.86 |
| smeared | 36 | 66.67 | 50.00 | 80.77 |
| spiralled | 36 | 16.67 | 25.00 | 40.00 |
| sprinkled | 36 | 25.00 | 19.44 | 58.06 |
| stained | 36 | 50.00 | 38.89 | 94.29 |
| stratified | 36 | 36.11 | 33.33 | — |
| striped | 36 | 8.33 | 19.44 | 34.29 |
| studded | 36 | 19.44 | 13.89 | 31.43 |
| swirly | 36 | 16.67 | 25.00 | 32.35 |
| veined | 36 | 30.56 | 27.78 | 54.55 |
| waffled | 36 | 2.78 | 2.78 | 11.43 |
| woven | 36 | 38.89 | 27.78 | 58.06 |
| wrinkled | 36 | 33.33 | 36.11 | 36.67 |
| zigzagged | 36 | 52.78 | 44.44 | — |

### ZeroShot-CS-s1  (tau_img = 100.0)

| class | n | patch_flip (%) | flip_vs_img (%) | patch_only_flip (%) |
|---|---|---|---|---|
| banded | 36 | 0.00 | — | — |
| blotchy | 36 | 0.00 | — | — |
| braided | 36 | 0.00 | — | — |
| bubbly | 36 | 0.00 | — | — |
| bumpy | 36 | 0.00 | — | — |
| chequered | 36 | 0.00 | — | — |
| cobwebbed | 36 | 0.00 | — | — |
| cracked | 36 | 0.00 | — | — |
| crosshatched | 36 | 0.00 | — | — |
| crystalline | 36 | 0.00 | — | — |
| dotted | 36 | 0.00 | — | — |
| fibrous | 36 | 0.00 | — | — |
| flecked | 36 | 0.00 | — | — |
| freckled | 36 | 0.00 | — | — |
| frilly | 36 | 0.00 | — | — |
| gauzy | 36 | 0.00 | — | — |
| grid | 36 | 0.00 | — | — |
| grooved | 36 | 0.00 | — | — |
| honeycombed | 36 | 0.00 | — | — |
| interlaced | 36 | 0.00 | — | — |
| knitted | 36 | 0.00 | — | — |
| lacelike | 36 | 0.00 | — | — |
| lined | 36 | 0.00 | — | — |
| marbled | 36 | 0.00 | — | — |
| matted | 36 | 0.00 | — | — |
| meshed | 36 | 0.00 | — | — |
| paisley | 36 | 0.00 | — | — |
| perforated | 36 | 0.00 | — | — |
| pitted | 36 | 0.00 | — | — |
| pleated | 36 | 0.00 | — | — |
| polka-dotted | 36 | 0.00 | — | — |
| porous | 36 | 0.00 | — | — |
| potholed | 36 | 0.00 | — | — |
| scaly | 36 | 0.00 | — | — |
| smeared | 36 | 0.00 | — | — |
| spiralled | 36 | 0.00 | — | — |
| sprinkled | 36 | 0.00 | — | — |
| stained | 36 | 0.00 | — | — |
| stratified | 36 | 0.00 | — | — |
| striped | 36 | 0.00 | — | — |
| studded | 36 | 0.00 | — | — |
| swirly | 36 | 0.00 | — | — |
| veined | 36 | 0.00 | — | — |
| waffled | 36 | 0.00 | — | — |
| woven | 36 | 0.00 | — | — |
| wrinkled | 36 | 0.00 | — | — |
| zigzagged | 36 | 0.00 | — | — |

### ZeroShot-CS-s2  (tau_img = 100.0)

| class | n | patch_flip (%) | flip_vs_img (%) | patch_only_flip (%) |
|---|---|---|---|---|
| banded | 36 | 0.00 | — | — |
| blotchy | 36 | 0.00 | — | — |
| braided | 36 | 0.00 | — | — |
| bubbly | 36 | 0.00 | — | — |
| bumpy | 36 | 0.00 | — | — |
| chequered | 36 | 0.00 | — | — |
| cobwebbed | 36 | 0.00 | — | — |
| cracked | 36 | 0.00 | — | — |
| crosshatched | 36 | 0.00 | — | — |
| crystalline | 36 | 0.00 | — | — |
| dotted | 36 | 0.00 | — | — |
| fibrous | 36 | 0.00 | — | — |
| flecked | 36 | 0.00 | — | — |
| freckled | 36 | 0.00 | — | — |
| frilly | 36 | 0.00 | — | — |
| gauzy | 36 | 0.00 | — | — |
| grid | 36 | 0.00 | — | — |
| grooved | 36 | 0.00 | — | — |
| honeycombed | 36 | 0.00 | — | — |
| interlaced | 36 | 0.00 | — | — |
| knitted | 36 | 0.00 | — | — |
| lacelike | 36 | 0.00 | — | — |
| lined | 36 | 0.00 | — | — |
| marbled | 36 | 0.00 | — | — |
| matted | 36 | 0.00 | — | — |
| meshed | 36 | 0.00 | — | — |
| paisley | 36 | 0.00 | — | — |
| perforated | 36 | 0.00 | — | — |
| pitted | 36 | 0.00 | — | — |
| pleated | 36 | 0.00 | — | — |
| polka-dotted | 36 | 0.00 | — | — |
| porous | 36 | 0.00 | — | — |
| potholed | 36 | 0.00 | — | — |
| scaly | 36 | 0.00 | — | — |
| smeared | 36 | 0.00 | — | — |
| spiralled | 36 | 0.00 | — | — |
| sprinkled | 36 | 0.00 | — | — |
| stained | 36 | 0.00 | — | — |
| stratified | 36 | 0.00 | — | — |
| striped | 36 | 0.00 | — | — |
| studded | 36 | 0.00 | — | — |
| swirly | 36 | 0.00 | — | — |
| veined | 36 | 0.00 | — | — |
| waffled | 36 | 0.00 | — | — |
| woven | 36 | 0.00 | — | — |
| wrinkled | 36 | 0.00 | — | — |
| zigzagged | 36 | 0.00 | — | — |

## Aggregate over PatchModPTA-CS seeds

- delta_mean = 5.02 pp
- rho_mean = -0.44

Per-seed:
  - PatchModPTA-CS-s1: delta = 5.44 pp, rho = -0.51
  - PatchModPTA-CS-s2: delta = 4.61 pp, rho = -0.37

DECISION: INCONCLUSIVE (delta_mean=5.02 pp, rho_mean=-0.44)

---

# PART B: Follow-up revalidation (fixed code, deterministic loader, seed 1, records_v2)

Generated from `outputs/records_v2/subset` (Task 5, plan `patch-proto-revalidation`). Subset means = `mean(correct)` over the 180 records per run. Classes (5-class closed set): bumpy(0), flecked(1), lacelike(2), lined(3), pitted(4) (`outputs/subset_classes.txt`). Fixed-code fingerprint: `gate_mode=="multi"` and `proto_alpha==1.0` on 180/180 PatchModPTA records; `fusion.tau_patch_proto` nested 2.5/5.0/10.0, top-level key absent. Sources: `.omo/evidence/wave2-gate.md`, `.omo/evidence/task-5-sweep-attest.txt`, `.omo/evidence/task-6-gate-check.txt`, `.omo/evidence/task-6-cluster-count.txt`, `/tmp/flip-subset.md`.

## Subset accuracy per run (seed 1, 180 records)

| run | method | tau | n | correct | subset mean acc (%) |
|---|---|---|---|---|---|
| PatchModPTA-fixed-tau2.5-s1 | PatchModPTA-fixed | 2.5 | 180 | 90 | **50.00** |
| PatchModPTA-fixed-tau5.0-s1 | PatchModPTA-fixed | 5.0 | 180 | 85 | **47.22** |
| PatchModPTA-fixed-tau10-s1 | PatchModPTA-fixed | 10.0 | 180 | 79 | **43.89** |
| PTA-fixed-s1 | PTA-fixed | n/a | 180 | 93 | **51.67** |
| ZeroShot-fixed-s1 | ZeroShot-fixed | n/a | 180 | 87 | **48.33** |

Independently recomputed (`.omo/evidence/task-6-gate-check.txt`, read-only script) and reproduced by the sweep attestation (`.omo/evidence/task-5-sweep-attest.txt` §3) and the runner result file (`outputs/result_subset_v2.txt`) exactly. The tau sweep is strictly monotonic decreasing in tau (spread 6.11 pp): the tau read is LIVE, the sweep has signal.

**Gate check (pre-registered, `.omo/evidence/gate-noise-sizing.md` §5)**: proceed iff `max over tau of (PatchModPTA-fixed-sub mean) >= PTA-fixed-sub mean`. max_tau = 50.00 (tau 2.5) < PTA-fixed-sub = 51.67 → gate **FAILS** (difference -1.67 pp, no tie). **STOP branch: Wave 3 NOT run.**

**Point-5 bar NOT testable**: `rho_mean > 0 AND delta_mean > 0` requires full-DTD per-seed PatchModPTA stream metrics (delta = last-half minus first-half accuracy; rho = Spearman of cluster growth vs last-half per-class accuracy). Those runs were Wave 3 scope and were skipped. The subset means above are the gate instrument only; rho/delta are NOT computed from subset records and the point-5 bar was never evaluated.

## Subset accuracy per class (seed 1, n=36 per class)

PatchModPTA-fixed-tau2.5 vs PTA-fixed vs ZeroShot-fixed (Δ vs PTA in pp) [`.omo/evidence/wave2-gate.md` §3.1]:

| class | PatchModPTA tau2.5 (%) | PTA (%) | ZeroShot (%) | Δ vs PTA (pp) |
|---|---|---|---|---|
| bumpy | 72.2 (26/36) | 80.6 (29/36) | 66.7 (24/36) | **-8.3** |
| flecked | 0.0 (0/36) | 2.8 (1/36) | 0.0 (0/36) | **-2.8** |
| lacelike | 69.4 (25/36) | 75.0 (27/36) | 77.8 (28/36) | **-5.6** |
| lined | 91.7 (33/36) | 88.9 (32/36) | 75.0 (27/36) | **+2.8** |
| pitted | 16.7 (6/36) | 11.1 (4/36) | 22.2 (8/36) | **+5.6** |

Full tau sweep per class (Δ vs PTA, pp):

| class | tau2.5 | tau5.0 | tau10 |
|---|---|---|---|
| bumpy | -8.3 | -8.3 | -8.3 |
| flecked | -2.8 | -2.8 | -2.8 |
| lacelike | -5.6 | -22.2 | -36.1 |
| lined | +2.8 | +5.6 | +0.0 |
| pitted | +5.6 | +5.6 | +8.3 |

3 of 5 classes worsened; lacelike collapses monotonically with tau.

## Cluster availability per class (true-side `proto_stats`, n=36 per class)

Counts = records of that class with true-side `n_clusters == 0` [`.omo/evidence/task-6-cluster-count.txt`]:

| run | bumpy | flecked | lacelike | lined | pitted | classes fully zero |
|---|---|---|---|---|---|---|
| PatchModPTA-fixed-tau2.5-s1 | 0/36 | 4/36 | 1/36 | 1/36 | 2/36 | 0 |
| PatchModPTA-fixed-tau5.0-s1 | 0/36 | 4/36 | 1/36 | 1/36 | 2/36 | 0 |
| PatchModPTA-fixed-tau10-s1 | 0/36 | 4/36 | 1/36 | 1/36 | 2/36 | 0 |
| PTA-fixed-s1 | 36/36* | 36/36* | 36/36* | 36/36* | 36/36* | 5 |
| ZeroShot-fixed-s1 | 36/36* | 36/36* | 36/36* | 36/36* | 36/36* | 5 |

\* PTA/ZeroShot `proto_stats` sides are null on all 180 records (patch term structurally absent by design). Every PatchModPTA class forms clusters on ≥32 of 36 samples; the gate miss is not attributable to missing clusters.

## Flip diagnostics summary (flips-v2, GT-aware, subset records)

Per-component predictions reconstructed from STORED logits (ProtoAlphaFusion formula, proto_alpha = 1.0). Full per-class tables in `outputs/hypothesis_findings.md` Part B §B.4. Class-name caveat: subset headers carry `classnames==[]`; CLI display order was full-DTD names, post-processed to the true subset classes (verified against the ZeroShot per-class column above).

| run | text-only acc (%) | +image (corr/reg, acc %, net) | +patch-alone (corr/reg, acc %, net) | +patch-with-image (corr/reg, acc %, net) |
|---|---|---|---|---|
| PTA-fixed-s1 | 48.33 (87/180) | 15/9 (62.50, **+6**) | masked 180 | masked 180 |
| PatchModPTA-fixed-tau2.5-s1 | 48.33 (87/180) | 12/6 (66.67, **+6**) | 17/39 (30.36, **-22**) | 2/6 (25.00, **-4**) |
| PatchModPTA-fixed-tau5.0-s1 | 48.33 (87/180) | 12/6 (66.67, **+6**) | 17/39 (30.36, **-22**) | 6/14 (30.00, **-8**) |
| PatchModPTA-fixed-tau10-s1 | 48.33 (87/180) | 12/6 (66.67, **+6**) | 18/39 (31.58, **-21**) | 8/22 (26.67, **-14**) |
| ZeroShot-fixed-s1 | 48.33 (87/180) | masked | masked 180 | masked 180 |

The image-EMA component carries the entire gain (+6 net in every run); patch-alone is a net detractor (-21 to -22); the fused patch contribution worsens with tau (-4/-8/-14).

**Part B bottom line**: gate FAILED (50.00 < 51.67), STOP branch, Wave 3 not run, point-5 bar NOT tested. The old INCONCLUSIVE verdict (Part A above, old code) is preserved as history and is superseded in context by the Part B gate failure for the fixed-code path.
