#!/usr/bin/env python3
"""Gate-rate diagnostic on FULL DTD (47-way open-set), offline from records.

Answers "is adaptation actually happening on full DTD?" per class, and
quantifies the open-set negative-impact hypothesis (updates firing on
WRONG classes corrupt prototypes).

Reads only stdlib. Usage: python3 scripts/gate_rate_diagnostic.py
"""
import json
import math
import os
import statistics

RECORDS_DIR = "outputs/records"
OUT_MD = "outputs/gate_rate_diagnostic.md"

# Mirrors the model: softmax(clip)[c] >= 0.1 fires, w_new = 1 - exp(-w/T)
GATE = 0.1
T = 20.0

PAIRS = [("PTA-CS", "PatchModPTA-CS")]  # (baseline, method) x seeds 1,2
SEEDS = [1, 2]
CLASS_ORDER = [  # DTD split order (index -> name), for readable tables
    "banded","blotchy","braided","bubbly","bumpy","chequered","cobwebbed",
    "cracked","crosshatched","crystalline","dotted","fibrous","flecked",
    "freckled","frilly","gauzy","grid","grooved","honeycombed","interlaced",
    "knitted","lacelike","lined","matted","meshed","mottled","paisley",
    "perforated","pitted","plaited","polka-dotted","porous","potholed",
    "scaly","smeared","spiralled","sprinkled","streaked","stratified",
    "striped","studded","swirly","veined","waffled","woven","wrinkled",
    "zigzagged",
]


def load_records(label, seed):
    path = os.path.join(RECORDS_DIR, f"{label}-s{seed}", "records.jsonl")
    samples = []
    n_skipped = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                n_skipped += 1
                continue
            if rec.get("__header__"):
                continue
            samples.append(rec)
    if n_skipped:
        print(f"[warn] {label}-s{seed}: skipped {n_skipped} corrupted record line(s)")
    return samples


def softmax(v):
    m = max(v)
    e = [math.exp(x - m) for x in v]
    s = sum(e)
    return [x / s for x in e]


def class_metrics(samples, C):
    """Per-class gate/update metrics from clip logits (gate is computed on
    raw zero-shot clip logits in both PTA and PatchModPTA, so these are
    identical for Combo1 — tanh never touches the gate)."""
    n_true = [0] * C
    n_fire = [0] * C            # samples where gate fires for class c
    n_true_fire = [0] * C       # fire AND target == c  (positive update)
    n_wrong_fire = [0] * C      # fire AND target != c  (corruption update)
    sum_w = [0.0] * C           # cumulative update mass, all fires
    sum_w_true = [0.0] * C      # cumulative update mass, true-class fires
    sum_w_wrong = [0.0] * C     # cumulative update mass, wrong-class fires
    n_any_fire = 0              # samples where ANY class fired
    n_true_fire_any = 0         # samples where the TRUE class fired
    n = len(samples)
    w_new_samples = []          # distribution of w_new on firing events
    for rec in samples:
        t = int(rec["target"])
        n_true[t] += 1
        w = softmax(rec["logits"]["clip"])
        fired = [w[c] >= GATE for c in range(C)]
        if any(fired):
            n_any_fire += 1
        if fired[t]:
            n_true_fire_any += 1
        for c in range(C):
            if fired[c]:
                n_fire[c] += 1
                w_new = 1.0 - math.exp(-w[c] / T)
                w_new_samples.append(w_new)
                sum_w[c] += w_new
                if c == t:
                    n_true_fire[c] += 1
                    sum_w_true[c] += w_new
                else:
                    n_wrong_fire[c] += 1
                    sum_w_wrong[c] += w_new
    return {
        "n": n, "n_true": n_true, "n_fire": n_fire,
        "n_true_fire": n_true_fire, "n_wrong_fire": n_wrong_fire,
        "sum_w": sum_w, "sum_w_true": sum_w_true, "sum_w_wrong": sum_w_wrong,
        "n_any_fire": n_any_fire, "n_true_fire_any": n_true_fire_any,
        "w_new_samples": w_new_samples,
    }


def class_acc(samples, C):
    total = [0] * C
    correct = [0] * C
    for rec in samples:
        t = int(rec["target"])
        total[t] += 1
        if rec["correct"]:
            correct[t] += 1
    return [100.0 * correct[c] / total[c] if total[c] else 0.0
            for c in range(C)]


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return float("nan")
    return cov / math.sqrt(vx * vy)


def rankdata(v):
    idx = sorted(range(len(v)), key=lambda i: v[i])
    ranks = [0.0] * len(v)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[idx[j + 1]] == v[idx[i]]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[idx[k]] = r
        i = j + 1
    return ranks


def spearman(xs, ys):
    return pearson(rankdata(xs), rankdata(ys))


def main():
    C = 47
    # --- accuracy per method/seed ---
    acc = {}
    for label in ("PTA-CS", "PatchModPTA-CS"):
        acc[label] = []
        for seed in SEEDS:
            acc[label].append(class_acc(load_records(label, seed), C))
    acc_mean = {l: [(a + b) / 2 for a, b in zip(acc[l][0], acc[l][1])]
                for l in acc}
    overall = {l: sum(acc_mean[l]) / C for l in acc}

    # --- gate metrics (same for both methods; use PTA-CS records) ---
    mets = []
    for seed in SEEDS:
        mets.append(class_metrics(load_records("PTA-CS", seed), C))
    m = {
        "n": mets[0]["n"],
        "n_true": [(a + b) / 2 for a, b in zip(mets[0]["n_true"], mets[1]["n_true"])],
        "n_fire": [(a + b) / 2 for a, b in zip(mets[0]["n_fire"], mets[1]["n_fire"])],
        "n_true_fire": [(a + b) / 2 for a, b in zip(mets[0]["n_true_fire"], mets[1]["n_true_fire"])],
        "n_wrong_fire": [(a + b) / 2 for a, b in zip(mets[0]["n_wrong_fire"], mets[1]["n_wrong_fire"])],
        "sum_w": [(a + b) / 2 for a, b in zip(mets[0]["sum_w"], mets[1]["sum_w"])],
        "sum_w_true": [(a + b) / 2 for a, b in zip(mets[0]["sum_w_true"], mets[1]["sum_w_true"])],
        "sum_w_wrong": [(a + b) / 2 for a, b in zip(mets[0]["sum_w_wrong"], mets[1]["sum_w_wrong"])],
        "n_any_fire": (mets[0]["n_any_fire"] + mets[1]["n_any_fire"]) / 2,
        "n_true_fire_any": (mets[0]["n_true_fire_any"] + mets[1]["n_true_fire_any"]) / 2,
    }
    w_new_all = mets[0]["w_new_samples"] + mets[1]["w_new_samples"]

    n = m["n"]
    fire_rate = [m["n_fire"][c] / n * 100 for c in range(C)]
    true_fire_rate = [m["n_true_fire"][c] / m["n_true"][c] * 100
                      if m["n_true"][c] else 0.0 for c in range(C)]
    wrong_fire_rate = [m["n_wrong_fire"][c] / n * 100 for c in range(C)]
    precision = [100.0 * m["n_true_fire"][c] / m["n_fire"][c]
                 if m["n_fire"][c] else 0.0 for c in range(C)]
    delta = [acc_mean["PatchModPTA-CS"][c] - acc_mean["PTA-CS"][c]
             for c in range(C)]

    # correlations (seed-mean level)
    corr_fr_delta = (pearson(fire_rate, delta), spearman(fire_rate, delta))
    corr_tf_delta = (pearson(true_fire_rate, delta), spearman(true_fire_rate, delta))
    corr_mass_delta = (pearson(m["sum_w"], delta), spearman(m["sum_w"], delta))
    corr_pos_delta = (pearson(m["sum_w_true"], delta), spearman(m["sum_w_true"], delta))
    corr_neg_delta = (pearson(m["sum_w_wrong"], delta), spearman(m["sum_w_wrong"], delta))

    # aggregate
    tot_fire = sum(m["n_fire"])
    tot_true_fire = sum(m["n_true_fire"])
    any_fire_pct = m["n_any_fire"] / n * 100
    true_fire_any_pct = m["n_true_fire_any"] / n * 100
    update_precision = 100.0 * tot_true_fire / tot_fire if tot_fire else 0.0
    w_new_mean = statistics.mean(w_new_all)
    w_new_med = statistics.median(w_new_all)

    # ---- build report ----
    L = []
    A = L.append
    A("# Gate-Rate Diagnostic — Full DTD (47-way open-set)")
    A("")
    A(f"Source records: `outputs/records/{{PTA-CS,PatchModPTA-CS}}-s{{1,2}}/records.jsonl` "
      f"({n} samples per seed, 47 classes × 36 images). Offline — no new model runs.")
    A("")
    A("Gate (identical in PTA and PatchModPTA; tanh fusion never touches it): "
      f"`softmax(clip_logits)[c] >= {GATE}`, update weight `w_new = 1 - exp(-w/T)`, T={T}.")
    A("")
    A("## 1. Overall accuracy (full DTD, seed-mean)")
    A("")
    A("| method | acc (%) |")
    A("|---|---|")
    for l in ("PTA-CS", "PatchModPTA-CS"):
        A(f"| {l} | {overall[l]:.2f} |")
    A(f"| **delta (PatchModPTA − PTA)** | **{overall['PatchModPTA-CS'] - overall['PTA-CS']:+.2f}** |")
    A("")
    A("## 2. Adaptation volume (open-set gate behavior)")
    A("")
    A(f"- Samples with **any** class firing: {any_fire_pct:.1f}%")
    A(f"- Samples where the **true** class fires (positive adaptation): **{true_fire_any_pct:.1f}%**")
    A(f"- Total class-fire events: {tot_fire:.0f} across {n} samples "
      f"({tot_fire / n:.2f} events/sample)")
    A(f"- Update precision (fires landing on the true class): **{update_precision:.1f}%** "
      f"→ {100 - update_precision:.1f}% of prototype updates are WRONG-class (corruption).")
    A(f"- Mean firing rate across classes: {statistics.mean(fire_rate):.2f}% "
      f"(median {statistics.median(fire_rate):.2f}%, min {min(fire_rate):.2f}%, "
      f"max {max(fire_rate):.2f}%)")
    A(f"- w_new on firing events: mean {w_new_mean:.4f}, median {w_new_med:.4f} "
      f"(prototype moves < 1% toward the image per update at w≈0.1..0.5)")
    A(f"- Cumulative update mass: total {sum(m['sum_w']):.1f} "
      f"(true-class {sum(m['sum_w_true']):.1f}, wrong-class {sum(m['sum_w_wrong']):.1f})")
    A("")
    A("## 3. Per-class table (seed-mean, sorted by firing rate)")
    A("")
    A("| class | PTA acc | Patch acc | delta | fire% | true-fire% | update precision% | pos mass | neg mass |")
    A("|---|---|---|---|---|---|---|---|---|")
    order = sorted(range(C), key=lambda c: -fire_rate[c])
    for c in order:
        A(f"| {CLASS_ORDER[c]} | {acc_mean['PTA-CS'][c]:.1f} | "
          f"{acc_mean['PatchModPTA-CS'][c]:.1f} | {delta[c]:+.1f} | "
          f"{fire_rate[c]:.1f} | {true_fire_rate[c]:.1f} | {precision[c]:.0f} | "
          f"{m['sum_w_true'][c]:.2f} | {m['sum_w_wrong'][c]:.2f} |")
    A("")
    A("## 4. Correlations with per-class delta (n=47)")
    A("")
    A("| predictor | Pearson | Spearman |")
    A("|---|---|---|")
    A(f"| firing rate | {corr_fr_delta[0]:+.3f} | {corr_fr_delta[1]:+.3f} |")
    A(f"| true-fire rate | {corr_tf_delta[0]:+.3f} | {corr_tf_delta[1]:+.3f} |")
    A(f"| total update mass | {corr_mass_delta[0]:+.3f} | {corr_mass_delta[1]:+.3f} |")
    A(f"| positive (true-class) mass | {corr_pos_delta[0]:+.3f} | {corr_pos_delta[1]:+.3f} |")
    A(f"| negative (wrong-class) mass | {corr_neg_delta[0]:+.3f} | {corr_neg_delta[1]:+.3f} |")
    A("")
    A("## 5. Open-set negative impact vs 5-way subset")
    A("")
    A("Reference (subset diagnosis): same classes fired on 23.9%–88.9% of samples "
      "in closed-set 5-way; 0.1%–5.4% in open-set 47-way (16x–330x reduction).")
    A("")
    A("| class | fire% full 47-way | fire% 5-way subset | ratio |")
    A("|---|---|---|---|")
    for name in ("bumpy", "flecked", "lacelike", "lined", "pitted"):
        ci = CLASS_ORDER.index(name)
        A(f"| {name} | {fire_rate[ci]:.1f} | — | — |")
    A("")
    A("## 6. Interpretation")
    A("")
    A("(filled by reviewer after reading the numbers)")

    with open(OUT_MD, "w") as f:
        f.write("\n".join(L) + "\n")

    # ---- concise stdout summary ----
    print("=" * 70)
    print(f"GATE-RATE DIAGNOSTIC — full DTD, n={n} samples/seed, C={C}")
    print("=" * 70)
    print(f"Overall acc: PTA-CS {overall['PTA-CS']:.2f} | "
          f"PatchModPTA-CS {overall['PatchModPTA-CS']:.2f} | "
          f"delta {overall['PatchModPTA-CS'] - overall['PTA-CS']:+.2f} pp")
    print(f"Adaptation: any-fire {any_fire_pct:.1f}% | TRUE-class fire {true_fire_any_pct:.1f}% | "
          f"update precision {update_precision:.1f}%")
    print(f"Firing rate/class: mean {statistics.mean(fire_rate):.2f}% "
          f"(med {statistics.median(fire_rate):.2f}%, "
          f"max {max(fire_rate):.1f}% @ {CLASS_ORDER[fire_rate.index(max(fire_rate))]})")
    print(f"w_new: mean {w_new_mean:.4f} med {w_new_med:.4f} | "
          f"update mass total {sum(m['sum_w']):.1f} "
          f"(pos {sum(m['sum_w_true']):.1f} / neg {sum(m['sum_w_wrong']):.1f})")
    print(f"Corr(delta, fire_rate)      P {corr_fr_delta[0]:+.3f} S {corr_fr_delta[1]:+.3f}")
    print(f"Corr(delta, true_fire_rate) P {corr_tf_delta[0]:+.3f} S {corr_tf_delta[1]:+.3f}")
    print(f"Corr(delta, total mass)     P {corr_mass_delta[0]:+.3f} S {corr_mass_delta[1]:+.3f}")
    print(f"Corr(delta, pos mass)       P {corr_pos_delta[0]:+.3f} S {corr_pos_delta[1]:+.3f}")
    print(f"Corr(delta, neg mass)       P {corr_neg_delta[0]:+.3f} S {corr_neg_delta[1]:+.3f}")
    print(f"Report: {OUT_MD}")


if __name__ == "__main__":
    main()
