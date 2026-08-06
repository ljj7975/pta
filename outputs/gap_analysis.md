# Gap Analysis — Selected Difficult Classes (bottom-5)

Selected by the pre-registered rule: mean PatchModPTA-CS per-class accuracy across seeds
(s1 + s2), ascending, bottom-5, ordered by `datasets.build_dataset('dtd').classnames`.
Selection produced by `scripts/analyze_records.py select --records outputs/records --k 5`
(no hand-picking).

Selected classes: `bumpy`, `flecked`, `lacelike`, `lined`, `pitted`

## Per-seed per-class accuracy (%)

| class | PatchModPTA-CS-s1 | PatchModPTA-CS-s2 | PTA-CS-s1 | PTA-CS-s2 | ZeroShot-CS-s1 | ZeroShot-CS-s2 |
|---|---|---|---|---|---|---|
| bumpy | 0.00 | 2.78 | 0.00 | 0.00 | 2.78 | 2.78 |
| flecked | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| lacelike | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| lined | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| pitted | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

## Mean across seeds + gaps (pp)

| class | PatchModPTA | PTA | PatchModPTA − PTA | ZeroShot | PatchModPTA − ZeroShot |
|---|---|---|---|---|---|
| bumpy | 1.39 | 0.00 | **+1.39** | 2.78 | **−1.39** |
| flecked | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| lacelike | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| lined | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| pitted | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 |

## Interpretation

All five selected classes sit at or near the accuracy floor: `flecked`, `lacelike`, `lined`,
and `pitted` are at **0.00% mean accuracy for every method and every seed** (neither PTA,
ZeroShot, nor PatchModPTA solves a single test image), and `bumpy` is the only class with
any signal (1.39% PatchModPTA, 2.78% ZeroShot). PatchModPTA wins vs PTA on `bumpy`
(+1.39 pp, its only nonzero gap) and ties on the other four; vs ZeroShot it **loses** on
`bumpy` (−1.39 pp) and ties everywhere else. Because the gaps are essentially zero, these
classes are best interpreted as an unsolved "hardest floor" subset rather than a set where
the methods differentiate — the closed-set T13 subset evaluation will be operating near the
ceiling of difficulty for all three methods.

Generated from `outputs/perclass_tables.md` (cross-method diff table) — values match the
`select` subcommand output exactly.
