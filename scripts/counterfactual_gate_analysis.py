#!/usr/bin/env python3
"""Offline counterfactual analysis of per-sample gates for the patch vote.

Reads the per-sample logit records written by patch_modulated_pta
(RECORD_DIR mode) and re-evaluates what the fused prediction WOULD have
been under different vote-gating policies, without touching the model or
the adaptation trajectory.

Policies evaluated on frozen logits (tau_text=1, tau_image_proto=80,
tau_patch_proto=10, tanh squash scale=1.0, matching the Combo1 run):

  baseline  current fusion: clip + 80*image + 10*tanh(patch)          (gate always 1)
  a_gate    agreement gate: patch muted (g=0) when patch.argmax != clip.argmax
  b_drop    agreement-drop: patch muted when clip and image already agree
            (patch kept only as a tie-breaker on clip/image disagreement)
  c_majority 2-of-3 vote among clip/image/patch; no-majority fallback =
            clip + 80*image (patch never decides a 3-way split alone)

Usage: python3 scripts/counterfactual_gate_analysis.py [records.jsonl]
"""

import json
import math
import sys
from collections import Counter, defaultdict

TAU_TEXT, TAU_IMG, TAU_PATCH, SCALE = 1.0, 80.0, 10.0, 1.0


def argmax(vec):
    return max(range(len(vec)), key=lambda i: vec[i])


def tanh_squash(vec):
    return [math.tanh(v / SCALE) for v in vec]


def fused_no_patch(clip, img):
    return [TAU_TEXT * c + TAU_IMG * i for c, i in zip(clip, img)]


def fused_with_patch(clip, img, patch, gate):
    out = fused_no_patch(clip, img)
    if gate:
        sq = tanh_squash(patch)
        out = [o + TAU_PATCH * g * s for o, g, s in zip(out, gate, sq)]
    return out


def majority_vote(clip_arg, img_arg, patch_arg, patch_all_zero):
    votes = [clip_arg, img_arg]
    if not patch_all_zero:
        votes.append(patch_arg)
    counts = Counter(votes)
    winner, n = counts.most_common(1)[0]
    if n >= 2:
        return winner, "majority"
    return None, "split"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else (
        "outputs/records_proxy_validation/Combo1-full-dtd-s1/records.jsonl"
    )

    rows = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("__header__"):
                continue
            rows.append(rec)

    n = len(rows)
    stats = {
        "baseline": {"acc": 0, "flips": 0},
        "a_gate": {"acc": 0, "flips": 0},
        "b_drop": {"acc": 0, "flips": 0},
        "c_majority": {"acc": 0, "flips": 0},
    }
    # agreement bookkeeping across all samples
    agree_clip_img = 0      # clip.argmax == image.argmax
    agree_clip_patch = 0    # patch agrees with clip
    patch_all_zero_cnt = 0
    split3 = 0
    recon_mismatch = 0      # reconstructed fused argmax != recorded pred
    # accuracy of each gate state (does muting/keeping patch help?)
    state_acc = defaultdict(lambda: [0, 0])  # (policy, state) -> [correct, total]
    for r in rows:
        clip = r["logits"]["clip"]
        img = r["logits"]["image_proto"]
        patch = r["logits"]["patch_proto"]
        target = r["target"]
        recorded_pred = r["pred"]

        clip_arg, img_arg, patch_arg = argmax(clip), argmax(img), argmax(patch)
        patch_zero = all(abs(p) < 1e-9 for p in patch)

        recon = argmax(fused_with_patch(clip, img, patch, [1.0] * len(clip)))
        if recon != recorded_pred:
            recon_mismatch += 1

        if clip_arg == img_arg:
            agree_clip_img += 1
        if patch_arg == clip_arg:
            agree_clip_patch += 1
        if patch_zero:
            patch_all_zero_cnt += 1

        # baseline: always-on patch
        pred = recon
        stats["baseline"]["acc"] += pred == target
        stats["baseline"]["flips"] += pred != recorded_pred

        # (a) agreement gate: mute patch when it votes against clip
        gate_a = [1.0 if patch_arg == clip_arg else 0.0] * len(clip)
        pred_a = argmax(fused_with_patch(clip, img, patch, gate_a))
        stats["a_gate"]["acc"] += pred_a == target
        stats["a_gate"]["flips"] += pred_a != recorded_pred
        state_acc[("a_gate", "agree" if patch_arg == clip_arg else "disagree")][0] += pred_a == target
        state_acc[("a_gate", "agree" if patch_arg == clip_arg else "disagree")][1] += 1

        # (b) agreement-drop: keep patch only when clip/image conflict
        keep = clip_arg != img_arg
        gate_b = [1.0 if keep else 0.0] * len(clip)
        pred_b = argmax(fused_with_patch(clip, img, patch, gate_b))
        stats["b_drop"]["acc"] += pred_b == target
        stats["b_drop"]["flips"] += pred_b != recorded_pred
        state_acc[("b_drop", "keep" if keep else "drop")][0] += pred_b == target
        state_acc[("b_drop", "keep" if keep else "drop")][1] += 1

        # (c) 2-of-3 majority
        winner, how = majority_vote(clip_arg, img_arg, patch_arg, patch_zero)
        if winner is None:
            split3 += 1
            pred_c = argmax(fused_no_patch(clip, img))
        else:
            pred_c = winner
        stats["c_majority"]["acc"] += pred_c == target
        stats["c_majority"]["flips"] += pred_c != recorded_pred

    print(f"records: {n}   target acc from summary.json: 47.104%")
    print(f"reconstructed fused argmax == recorded pred: {n - recon_mismatch}/{n} "
          f"({100.0 * (n - recon_mismatch) / n:.2f}%)")
    print(f"clip.argmax == image.argmax: {agree_clip_img}/{n} "
          f"({100.0 * agree_clip_img / n:.1f}%)")
    print(f"patch.argmax == clip.argmax: {agree_clip_patch}/{n} "
          f"({100.0 * agree_clip_patch / n:.1f}%)")
    print(f"patch logits all zero (no patch evidence): {patch_all_zero_cnt}/{n} "
          f"({100.0 * patch_all_zero_cnt / n:.1f}%)")
    print(f"3-way split (no majority, fallback used): {split3}/{n} "
          f"({100.0 * split3 / n:.1f}%)")
    print()
    print(f"{'policy':<14}{'acc':>8}{'delta':>8}{'flips':>8}")
    for name, s in stats.items():
        acc = 100.0 * s["acc"] / n
        print(f"{name:<14}{acc:>7.2f}%{acc - 47.104:>+7.2f}{s['flips']:>8}")
    print()
    print("state breakdown (correct/total -> acc):")
    for (policy, state), (corr, tot) in sorted(state_acc.items()):
        print(f"  {policy:<10} {state:<9} {corr:>4}/{tot:<4} {100.0 * corr / tot:6.2f}%")


if __name__ == "__main__":
    main()
