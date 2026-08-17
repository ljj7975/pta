#!/usr/bin/env python3
"""
proto_viz_tau25.py — Faithful replay + visualisation of the
PatchModPTA-fixed-tau2.5-s1 DTD subset run (seed 1, 5 difficult classes).

WHY
---
The Wave-2 subset run ended NEGATIVE: at tau_patch_proto=2.5 the patch term
does *not* help (50.00% overall; flecked 0.0%, pitted 16.67% vs lined 91.67%).
The records only store logits + cluster-count stats — no patch pixels, cluster
centers, or appearance — so this script REGENERATES the prototypes by exactly
replaying the recorded run (deterministic seeds, same encoder, same loader,
verbatim PatchModulatedPTAAdapter.run loop order) and produces:

  outputs/proto_viz_tau2.5/replay_check.json      — per-batch fidelity vs records
  outputs/proto_viz_tau2.5/<classname>/proto_final.png   — end-of-run prototype montage
  outputs/proto_viz_tau2.5/<classname>/sample_<batch>.png — 5 sample scoring pages
  outputs/proto_viz_tau2.5/summary.json           — per-class end state
  outputs/proto_viz_tau2.5/resolved_config.json   — exact config from record header

FIDELITY GATE
-------------
Every one of the 180 batches is compared field-by-field against
records.jsonl (target, pred, correct, proto_stats true/pred n_images+n_clusters,
and all four logit vectors atol=1e-4).  The script prints
"REPLAY FIDELITY: N/180 batches match"; on a full run with N < 180 it prints the
first 5 mismatches and exits with code 1 — the figures are only trustworthy
with an exact match.

REUSED RENDERING HELPERS
------------------------
All image rendering comes from tests/debug_gaussian_patch.py (imported as dgp):
_denorm, _assignment_overlay, _crops_strip, _assign_all_patches,
_filter_heatmap_overlay, _rgb, _CMAP, GRID=14, PSIZE=16, IMPX=224.

USAGE
-----
  python tests/proto_viz_tau25.py \
      --seed 1 --samples-per-class 5 \
      --output-dir outputs/proto_viz_tau2.5 \
      --record outputs/records_v2/subset/PatchModPTA-fixed-tau2.5-s1/records.jsonl \
      --class-file outputs/subset_classes.txt --data-root ./data

Hidden flag --max-batches N caps the replay loop (smoke tests only; default 0 =
unlimited).  GPU required (the shared clip_classifier/get_clip_logits helpers
are hardcoded to .cuda() and the ViT-B/16 surgery forward is impractically slow
on CPU) — launch via srun.

Deterministic setup is identical to runner.py lines 174-182, and the replay loop
is a verbatim replication of models/patch_modulated_pta.py PatchModulatedPTAAdapter.run
(lines 222-347) with visualisation hooks.  The algorithm is NOT modified in any
way.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from PIL import Image as PilImage

# Repo root + tests/ on sys.path (so `import debug_gaussian_patch as dgp` works
# from either the repo root or the tests/ directory).
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests"))

import debug_gaussian_patch as dgp  # noqa: E402  — rendering helpers

from encoder import create_encoder_instance  # noqa: E402
from models.patch_modulated_pta import (  # noqa: E402
    build as build_adapter,
    _update_text_features_with_quality,
)
from utils import cls_acc, get_clip_logits  # noqa: E402
from utils.clip_inference import _safe_normalize, clip_classifier  # noqa: E402
from utils.data import build_subset_test_data_loader  # noqa: E402
from models.patch_level.gaussian_patch import _augment_image  # noqa: E402

# 0-based occurrence ranks within each class's 36-sample stream (the 1st, 9th,
# 18th, 27th, 36th occurrence) — the 5 sample pages per class.
SAMPLE_RANKS = [0, 8, 17, 26, 35]


# ── small helpers ─────────────────────────────────────────────────────────────

def _cpu_details(d: dict) -> dict:
    """Recursively move a details-dict (from compute_patch_logits) to CPU.

    Keys are per-prototype tensors plus `group_members` (a list of index
    tensors).  Anything else is passed through untouched.
    """
    out = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().cpu()
        elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
            out[k] = [t.detach().cpu() for t in v]
        else:
            out[k] = v
    return out


def _crop_rep_patch(img_tensor: torch.Tensor, patch_idx: int,
                    cell: int = 32, border=None) -> np.ndarray:
    """Crop the patch at *patch_idx* from a denormed image, resized to cell.

    Mirrors debug_gaussian_patch._crops_strip's rep-patch block: 2px border in
    the cluster colour when *border* is given.
    """
    img = dgp._denorm(img_tensor.squeeze(0))
    r, c = int(patch_idx) // dgp.GRID, int(patch_idx) % dgp.GRID
    raw = img[r * dgp.PSIZE:(r + 1) * dgp.PSIZE, c * dgp.PSIZE:(c + 1) * dgp.PSIZE]
    thumb = np.array(PilImage.fromarray(raw).resize((cell, cell), PilImage.NEAREST))
    if border is not None:
        thumb[:2, :] = border
        thumb[-2:, :] = border
        thumb[:, :2] = border
        thumb[:, -2:] = border
    return thumb


def _desaturate(block: np.ndarray) -> np.ndarray:
    """Desaturate + dim a canvas block (for FILTERED prototype rows)."""
    lum = block.mean(axis=-1, keepdims=True).astype(np.float32)
    out = (0.55 * lum + 0.15 * block.astype(np.float32)).clip(0, 255)
    return out.astype(np.uint8)


# ── proto_final montage ───────────────────────────────────────────────────────

def render_proto_final(c: int, class_name: str, state: dict,
                       class_imgs: list, acc: float,
                       appearance_min_weight: float, cls_dir: Path) -> Path:
    """End-of-run prototype montage for one class.

    One row per cluster k:
      - bordered rep-patch crops from top_rep_patches[k] (crop patch_idx from
        class_imgs[c][image_idx]), 32px cells, cluster-colour border.
      - left label `C{k} n=<appearance> w=<appearance/n_images:.3f> var=<mean var>`.
    Rows whose appearance weight falls below *appearance_min_weight* (i.e. the
    prototypes zeroed by the appearance filter during scoring) are desaturated,
    dimmed and flagged "[FILTERED]" — they contributed nothing to any score.
    """
    K = int(state["centers"].shape[0])
    n_images = int(state["n_images"])
    appearance = state["appearance"].cpu().numpy()
    variance = state["variance"].cpu()
    top_rep = list(state.get("top_rep_patches", []))

    cell = 32
    row_h = cell + 6
    n_rep = 5
    width = n_rep * (cell + 2) + 8
    canvas = np.full((K * row_h, width, 3), 255, dtype=np.uint8)

    for k in range(K):
        ry = k * row_h
        entries = top_rep[k] if k < len(top_rep) else []
        border = (dgp._rgb(k) * 255).astype(np.uint8)
        for j, (pidx, iidx, view_idx, _sim) in enumerate(entries[:n_rep]):
            thumb = np.full((cell, cell, 3), 235, dtype=np.uint8)
            if pidx >= 0 and iidx >= 0 and iidx < len(class_imgs):
                view_list = class_imgs[iidx]
                view_idx_safe = min(view_idx, len(view_list) - 1)
                thumb = _crop_rep_patch(view_list[view_idx_safe], pidx, cell=cell, border=border)
            x = 4 + j * (cell + 2)
            canvas[ry + 3:ry + 3 + cell, x:x + cell] = thumb
        if appearance[k] / max(n_images, 1) < appearance_min_weight:
            canvas[ry:ry + row_h] = _desaturate(canvas[ry:ry + row_h])

    fig, ax = plt.subplots(
        1, 1, figsize=(max(7, width / 100), max(3, K * row_h / 100))
    )
    ax.imshow(canvas)
    for k in range(K):
        y_frac = (k * row_h + row_h / 2) / canvas.shape[0]
        var_mean = float(variance[k].mean())
        w_k = appearance[k] / max(n_images, 1)
        filt = w_k < appearance_min_weight
        label = "C{k} n={n:.0f} w={w:.3f} var={var:.4f}".format(
            k=k, n=appearance[k], w=w_k, var=var_mean)
        if filt:
            label += "  [FILTERED]"
        ax.text(-0.008, 1 - y_frac, label, transform=ax.transAxes,
                fontsize=8, va="center", ha="right",
                color="dimgray" if filt else dgp._CMAP[k % 20][:3],
                fontweight="bold" if filt else "normal")
    ax.axis("off")
    ax.set_title(
        "{name}  |  n_images={ni}  n_clusters={nk}  acc={acc:.2f}%  |  "
        "appearance_min_weight={amw}".format(
            name=class_name, ni=n_images, nk=K, acc=acc, amw=appearance_min_weight),
        fontsize=11, fontweight="bold",
    )

    cls_dir.mkdir(parents=True, exist_ok=True)
    path = cls_dir / "proto_final.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


# ── per-sample scoring page ───────────────────────────────────────────────────

def render_sample_page(batch_idx: int, sel: dict, rec: dict,
                       classnames: list, encoder, fusion, out_dir: Path) -> Path:
    """One scoring page for a selected sample of class c.

    Panel (a)  cluster-assignment overlay on the original image (argmax vs the
               post-update class state) with the filter keep-mask graying.
    Panel (b)  filter keep-mask overlay (green=kept / red=filtered) from
               state["keep_mask"].
    Panel (c)  patch_proto_logits bar chart + quality_gate.
    Panel (d)  text annotation: target/pred, correct/wrong, conf (text), and
               final = tau_text*clip + tau_img*img_proto + tau_patch*patch.
    Row 2      patch crops grouped per cluster via the one-to-one scoring
               details (proto_group_idx / group_members) labelled
               `C{k} (n=, app=, w=)`.

    *sel* carries: class_idx, occ (occurrence rank+1), details_c (CPU details
    from compute_patch_logits at this batch, i.e. pre-update scoring), img_cpu
    (this batch's image), was_gated (was the class updated this batch?),
    state_after (frozen snapshot of the class state after this batch).
    """
    c = sel["class_idx"]
    class_name = classnames[c]
    img_cpu = sel["img_cpu"]
    state = sel["state_after"]
    details_c = sel["details_c"]
    K = int(state["centers"].shape[0])

    img = dgp._denorm(img_cpu.squeeze(0))
    assign, sims = (dgp._assign_all_patches(img_cpu, state, encoder) if K > 0
                    else (None, None))
    km_np = state["keep_mask"].cpu().numpy() if state.get("keep_mask") is not None else None

    overlay = dgp._assignment_overlay(img, assign, K, keep_mask=km_np)
    filter_overlay = (dgp._filter_heatmap_overlay(img, state["keep_mask"])
                      if km_np is not None else img.copy())

    # Crops strip from the one-to-one scoring details (pre-update bank, K_pre).
    strip = None
    K_pre = 0
    if details_c is not None:
        K_pre = int(details_c["best_per_proto"].numel())
        strip = dgp._crops_strip(img, assign, K_pre, sims=sims,
                                 keep_mask=None, show_rep=False,
                                 proto_details=details_c)

    fig = plt.figure(figsize=(18, 11))
    gs = GridSpec(2, 4, figure=fig, height_ratios=[1.0, 1.35],
                  hspace=0.45, wspace=0.30)

    # ── (a) cluster-assignment overlay ──────────────────────────────────────
    ax00 = fig.add_subplot(gs[0, 0])
    ax00.imshow(overlay)
    ax00.set_title(
        "cluster overlay vs state after this sample\n"
        "K={k} clusters   gated={g}".format(k=K, g=sel["was_gated"]),
        fontsize=9,
    )
    ax00.axis("off")
    if K > 0:
        handles = [mpatches.Patch(color=dgp._CMAP[k % 20][:3], label="C{}".format(k))
                   for k in range(min(K, 8))]
        ax00.legend(handles=handles, loc="lower right", fontsize=6,
                    ncol=min(K, 4), framealpha=0.75)

    # ── (b) filter keep-mask overlay ────────────────────────────────────────
    ax01 = fig.add_subplot(gs[0, 1])
    ax01.imshow(filter_overlay)
    kept = int(state["keep_mask"].sum().item()) if km_np is not None else None
    mask_src = "this sample" if sel["was_gated"] else "last absorbed sample"
    ax01.set_title(
        "filter keep-mask overlay\n(green=kept red=filtered)\n"
        "kept={k}/196   mask from: {src}".format(k=kept, src=mask_src)
        if kept is not None else "[no filter mask stored]",
        fontsize=9,
    )
    ax01.axis("off")

    # ── (c) patch_proto_logits bar chart + quality_gate ─────────────────────
    ax02 = fig.add_subplot(gs[0, 2])
    vals = np.asarray(rec["logits"]["patch_proto"], dtype=np.float32)
    colors = ["tab:orange" if i == c else "tab:gray"
              for i in range(len(classnames))]
    bars = ax02.bar(range(len(classnames)), vals, color=colors)
    ax02.set_xticks(range(len(classnames)))
    ax02.set_xticklabels(classnames, rotation=30, ha="right", fontsize=8)
    ax02.set_ylabel("patch_proto_logit", fontsize=8)
    ax02.axhline(0, color="black", linewidth=0.5, linestyle="--")
    qg = rec["quality_gate"] if rec["quality_gate"] is not None else 0.0
    ax02.set_title("patch_proto_logits   quality_gate={q:.4f}".format(q=qg),
                   fontsize=9)
    vmax = max(abs(vals)) if vals.size else 0.0
    for bar, v in zip(bars, vals):
        if abs(v) > 1e-6:
            ax02.text(bar.get_x() + bar.get_width() / 2,
                      v + vmax * 0.02 + 1e-9,
                      "{:.3f}".format(v), ha="center", va="bottom", fontsize=7)

    # ── (d) text annotation ─────────────────────────────────────────────────
    ax03 = fig.add_subplot(gs[0, 3])
    ax03.axis("off")
    tau_t = float(fusion.tau_text)
    tau_i = float(fusion.tau_image_proto)
    tau_p = float(fusion.tau_patch_proto)
    verdict = "CORRECT" if rec["correct"] else "WRONG"
    txt = (
        "class        : {cn}\n"
        "global batch : {b}  (occurrence #{o})\n"
        "target       : {t}\n"
        "pred         : {p}   [{v}]\n"
        "conf (text)  : {conf:.4f}\n"
        "final = {tt:g}*clip + {ti:g}*img_proto + {tp:g}*patch\n"
        "gated for {cn}: {g}\n"
        "K before={kpre}  K after={k}"
    ).format(
        cn=class_name, b=batch_idx, o=sel["occ"],
        t=classnames[rec["target"]], p=classnames[rec["pred"]],
        v=verdict, conf=rec["conf"], tt=tau_t, ti=tau_i, tp=tau_p,
        g=sel["was_gated"], kpre=K_pre, k=K,
    )
    ax03.text(0.02, 0.97, txt, transform=ax03.transAxes,
              fontsize=9, va="top", fontfamily="monospace")

    # ── row 2: crops strip with C{k} (n=, app=, w=) labels ──────────────────
    ax1 = fig.add_subplot(gs[1, :])
    if strip is not None:
        ax1.imshow(strip)
        row_h = 32 + 6
        app_pre = details_c.get("_appearance_pre")
        n_img_pre = details_c.get("_n_images_pre", 0)
        proto_group_idx = details_c["proto_group_idx"]
        group_members = details_c["group_members"]
        for k in range(K_pre):
            y_frac = (k * row_h + row_h / 2) / strip.shape[0]
            gidx = int(proto_group_idx[k].item())
            if gidx >= 0 and gidx < len(group_members):
                npatches = int(group_members[gidx].numel())
            else:
                npatches = 0
            app_raw = int(round(float(app_pre[k]))) if app_pre is not None else 0
            w = (app_raw / max(n_img_pre, 1)) if n_img_pre else 0.0
            ax1.text(
                -0.008, 1 - y_frac,
                "C{k} (n={n}, app={a}/{ni}, w={w:.2f})".format(
                    k=k, n=npatches, a=app_raw, ni=n_img_pre, w=w),
                transform=ax1.transAxes, fontsize=7, va="center", ha="right",
                color=dgp._CMAP[k % 20][:3],
            )
        ax1.set_title(
            "patch crops of this sample grouped by cluster — one-to-one scoring "
            "against the PRE-update bank (K_pre={k})".format(k=K_pre),
            fontsize=9,
        )
    else:
        ax1.text(0.5, 0.5, "no prototypes yet at scoring time (K_pre = 0)",
                 ha="center", va="center", fontsize=10)
        ax1.set_title("patch crops: K_pre = 0 (empty bank at scoring time)",
                      fontsize=9)
    ax1.axis("off")

    fig.suptitle("{cn} | sample page".format(cn=class_name),
                 fontsize=12, fontweight="bold")
    cls_dir = out_dir / class_name
    cls_dir.mkdir(parents=True, exist_ok=True)
    path = cls_dir / "sample_{b}.png".format(b=batch_idx)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


# ── records / fidelity ────────────────────────────────────────────────────────

def load_records(path: str):
    """Parse records.jsonl → (header dict, [sample dicts] sorted by batch_idx)."""
    header = None
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("__header__"):
                header = obj
            else:
                records.append(obj)
    if header is None:
        raise ValueError("No __header__ line found in {}".format(path))
    records.sort(key=lambda r: r["batch_idx"])
    return header, records


def compare_fidelity(replay, orig, total: int):
    """Field-by-field comparison of the first *total* batches.

    Compared fields: target, pred, correct, proto_stats.true.pred
    {n_images, n_clusters}, and the four logit vectors (allclose atol=1e-4).
    Returns (matched_batches, [mismatch dicts]).
    """
    mismatches = []
    matched = 0
    for i in range(total):
        rec = replay[i]
        o = orig[i]
        bad = []
        if rec["target"] != o["target"]:
            bad.append(("target", o["target"], rec["target"]))
        if rec["pred"] != o["pred"]:
            bad.append(("pred", o["pred"], rec["pred"]))
        if rec["correct"] != o["correct"]:
            bad.append(("correct", o["correct"], rec["correct"]))
        for side in ("true", "pred"):
            for key in ("n_images", "n_clusters"):
                got = rec["proto_stats"][side][key]
                exp = o["proto_stats"][side][key]
                if got != exp:
                    bad.append(("proto_stats.{}.{}".format(side, key), exp, got))
        for name in ("clip", "image_proto", "patch_proto", "final"):
            got = np.asarray(rec["logits"][name], dtype=np.float64)
            exp = np.asarray(o["logits"][name], dtype=np.float64)
            if not np.allclose(got, exp, atol=1e-4):
                bad.append(("logits.{}".format(name), exp.tolist(), got.tolist()))
        if bad:
            for field, exp, got in bad:
                mismatches.append({
                    "batch_idx": i, "field": field,
                    "expected": exp, "got": got,
                })
        else:
            matched += 1
    return matched, mismatches


def write_summary_json(out_dir: Path, classnames: list, states: list,
                       per_class_acc: dict, per_class_total: dict,
                       total: int, final_acc: float, seed: int) -> dict:
    """Per-class end state: n_images, n_clusters, acc, per-cluster app_w."""
    per = {}
    for c in range(len(classnames)):
        app = states[c]["appearance"].cpu().tolist()
        n_img = int(states[c]["n_images"])
        per["class_{}".format(c)] = {
            "classname": classnames[c],
            "n_images": n_img,
            "n_clusters": int(states[c]["centers"].shape[0]),
            "acc": round(per_class_acc[c], 4),
            "n_total": per_class_total[c],
            "app_w": [round(a / max(n_img, 1), 4) for a in app],
        }
    summary = {
        "method": "proto_viz_tau2.5-replay",
        "dataset": "dtd",
        "seed": seed,
        "total": total,
        "acc": round(final_acc, 4),
        "per_class": per,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def crosscheck_record_summary(record_summary_path: str, classnames: list,
                              states: list, per_class_acc: dict,
                              final_acc: float, out_dir: Path) -> None:
    """Print per-class acc + end-state vs the record run's summary.json."""
    with open(record_summary_path, "r") as f:
        s = json.load(f)
    print("\nPer-class end state (replay vs records summary):")
    for c in range(len(classnames)):
        rec = s["per_class"]["class_{}".format(c)]
        ni = int(states[c]["n_images"])
        nk = int(states[c]["centers"].shape[0])
        flag = ""
        if (ni != rec["n_images_end"] or nk != rec["n_clusters_end"]
                or abs(per_class_acc[c] - rec["acc"]) > 1e-6):
            flag = "   <-- MISMATCH"
        print(
            "  class_{c} {cn:<10} acc={a:6.2f}%  n_images={ni}  n_clusters={nk}"
            "   (records: acc={ra:.2f}%  n_images={rni}  n_clusters={rnk}){flag}".format(
                c=c, cn=classnames[c], a=per_class_acc[c], ni=ni, nk=nk,
                ra=rec["acc"], rni=rec["n_images_end"], rnk=rec["n_clusters_end"],
                flag=flag,
            )
        )
    print("  overall acc = {a:.2f}%   (records: {r:.2f}%)".format(
        a=final_acc, r=s["acc"]))


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Replay + visualise PatchModPTA-fixed-tau2.5-s1 DTD subset run.")
    p.add_argument("--seed", type=int, default=1,
                   help="Loader/adapter seed (default: 1, matches the record run).")
    p.add_argument("--samples-per-class", type=int, default=5,
                   help="Number of sample pages per class (default: 5).")
    p.add_argument("--output-dir", default="outputs/proto_viz_tau2.5",
                   help="Root directory for replay outputs.")
    p.add_argument("--record",
                   default="outputs/records_v2/subset/PatchModPTA-fixed-tau2.5-s1/records.jsonl",
                   help="Ground-truth records.jsonl (fidelity reference + config source).")
    p.add_argument("--class-file", default="outputs/subset_classes.txt",
                   help="One classname per line; defines label order 0..C-1.")
    p.add_argument("--data-root", default="./data",
                   help="Dataset root passed to the loader.")
    p.add_argument("--max-batches", type=int, default=0,
                   help="Hidden: cap the replay loop (smoke tests only; 0 = unlimited).")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Deterministic setup — verbatim runner.py lines 174-182 ─────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    # No recording during the replay (adapter-side utils are env-gated no-ops).
    os.environ["RECORD_DIR"] = ""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}".format(device))
    print("Loading encoder: clip_surgery / ViT-B/16")

    encoder = create_encoder_instance("clip_surgery", model_type="ViT-B/16",
                                      device=device)
    preprocess = encoder.preprocess

    # ── Class-file label order (source of truth for labels 0..C-1) ─────────
    with open(args.class_file, "r") as f:
        classnames = [ln.strip() for ln in f
                      if ln.strip() and not ln.strip().startswith("#")]
    C = len(classnames)
    print("Classes ({}): {}".format(C, classnames))
    assert classnames == ["bumpy", "flecked", "lacelike", "lined", "pitted"], (
        "Unexpected class order in {}: {}".format(args.class_file, classnames))

    # ── Deterministic subset loader (same as runner.py --class-file) ───────
    test_loader, _, template = build_subset_test_data_loader(
        "dtd", args.data_root, preprocess,
        class_file=args.class_file, seed=args.seed,
    )

    # ── Text features (same as runner.py) ──────────────────────────────────
    text_embeddings = clip_classifier(classnames, template, encoder)  # [D, C]

    # ── Exact config from the record header; dumped verbatim ───────────────
    header, _orig_records = load_records(args.record)
    cfg = header["resolved_config"]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "resolved_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print("Resolved config loaded from record header (dumped to {})".format(
        out_dir / "resolved_config.json"))

    # ── Adapter + state init (verbatim from adapter.run lines 206-217) ─────
    adapter = build_adapter(cfg)
    refine_feature = text_embeddings.t().float()          # [C, D]
    target_prototype = adapter.image_level.init_state(refine_feature)
    states = adapter.patch_level.init_state(refine_feature)
    if cfg.get("patch_level", {}).get("patch_filter_mode", "none") != "none":
        adapter.patch_level.set_text_context(text_embeddings, encoder, device)

    # ── Loop params (same reads as adapter.run) ─────────────────────────────
    _pl_cfg = cfg.get("patch_level", {})
    _il_cfg = cfg.get("image_level", {})
    conf_thresh = float(_pl_cfg.get("conf_threshold", cfg.get("conf_threshold", 0.5)))
    conf_source = str(cfg.get("conf_source", "text"))
    multi_gate = bool(_pl_cfg.get("multi_gate", cfg.get("multi_gate", False)))
    alpha_pta = float(_il_cfg.get("alpha", cfg.get("alpha", 0.01)))
    T_pta = float(_il_cfg.get("T", cfg.get("T", 20.0)))
    appearance_min_weight = float(_pl_cfg.get("appearance_min_weight", 0.0))
    aug_copies = int(_pl_cfg.get("aug_copies", 0))
    assert multi_gate, "resolved_config must have multi_gate=True (record run did)"
    assert conf_source == "text", "resolved_config must have conf_source=text (record run did)"

    sample_ranks = SAMPLE_RANKS[:max(args.samples_per_class, 0)]

    # ── Replay state ─────────────────────────────────────────────────────────
    class_imgs = [[] for _ in range(C)]   # per-class absorbed-image streams (CPU)
    occ = [0] * C                          # per-class occurrence counter
    replay_records = []                    # pre-update capture (mirrors write_record)
    selected = {}                          # global batch idx -> viz inputs

    with torch.no_grad():
        for i, (images, target) in enumerate(test_loader):
            if args.max_batches > 0 and i >= args.max_batches:
                break
            if isinstance(images, list):
                images = torch.cat(images, dim=0).to(device)
            else:
                images = images.to(device)
            target = target.to(device)
            tgt = int(target.item())

            occ[tgt] += 1
            is_selected = (occ[tgt] - 1) in sample_ranks

            # 1) CLIP forward
            image_features, clip_logits, _, _, _ = get_clip_logits(
                images, encoder, text_embeddings
            )
            feat = image_features.squeeze(0).float()
            feat_norm = _safe_normalize(feat)

            # 2) Patch-level Gaussian scoring + quality gate.
            #    NO target_class_idx → scoring is UNFILTERED (surgery filter
            #    only applies inside update_state).  return_details=True keeps
            #    want_details=True — the same details stats_enabled already
            #    forces, so values are identical to the recorded run.
            patch_proto_logits, quality_gate, details = (
                adapter.patch_level.compute_patch_logits(
                    images, encoder, states, return_details=True
                )
            )

            # Stash scoring details (CPU) for selected samples BEFORE any
            # state mutation — details reference the pre-update bank.
            if is_selected:
                dc = details.get(tgt)
                if dc is not None:
                    dc = _cpu_details(dc)
                    dc["_appearance_pre"] = states[tgt]["appearance"].detach().cpu()
                    dc["_n_images_pre"] = int(states[tgt]["n_images"])
                selected[i] = {
                    "class_idx": tgt,
                    "occ": occ[tgt],
                    "details_c": dc,
                    "img_cpu": None,
                    "was_gated": False,
                    "state_after": None,
                }

            # 3) Image-level prototype update (with quality modulation)
            soft_logits = F.softmax(clip_logits, dim=-1)
            refine_feature, target_prototype = _update_text_features_with_quality(
                image_features,
                soft_logits.half(),
                refine_feature,
                target_prototype,
                alpha=alpha_pta,
                T=T_pta,
                quality_gate=quality_gate,
                quality_modulation=adapter.quality_modulation,
            )

            # 4) Image-level proto logits
            image_proto_logits = image_features.half() @ refine_feature.half().T  # [1, C]

            # 5) Three-way fusion
            final_logits = adapter.fusion.forward(
                clip_logits,
                image_proto_logits,
                patch_proto_logits,
                quality_gate=quality_gate,
            )

            # 6) Confidence gate source (conf_source="text")
            if conf_source == "text":
                gate_logits = clip_logits
            elif conf_source == "image":
                gate_logits = image_proto_logits
            elif conf_source == "pta":
                gate_logits = clip_logits + adapter.fusion.tau_image_proto * image_proto_logits
            elif conf_source == "full":
                gate_logits = final_logits
            else:
                raise ValueError("Unknown conf_source: {}".format(conf_source))

            acc = cls_acc(final_logits, target)

            # 7) Capture replay record — same point adapter.run writes records
            #    (proto_stats read from states BEFORE this batch's update).
            pred_cls = int(final_logits.argmax(dim=-1).item())
            replay_records.append({
                "batch_idx": i,
                "target": tgt,
                "pred": pred_cls,
                "correct": bool(acc),
                "conf": float(F.softmax(gate_logits, dim=-1).max().item()),
                "quality_gate": (float(quality_gate.item())
                                 if torch.is_tensor(quality_gate) else None),
                "gate_mode": "multi" if multi_gate else "single",
                "proto_alpha": 1.0,
                "logits": {
                    "clip": clip_logits.squeeze(0).float().cpu().tolist(),
                    "image_proto": image_proto_logits.squeeze(0).float().cpu().tolist(),
                    "patch_proto": patch_proto_logits.squeeze(0).float().cpu().tolist(),
                    "final": final_logits.squeeze(0).float().cpu().tolist(),
                },
                "proto_stats": {
                    "true": {
                        "n_images": int(states[tgt]["n_images"]),
                        "n_clusters": int(states[tgt]["centers"].shape[0]),
                    },
                    "pred": {
                        "n_images": int(states[pred_cls]["n_images"]),
                        "n_clusters": int(states[pred_cls]["centers"].shape[0]),
                    },
                },
            })

            # 8) Multi-gate update — image appended to each gated class's
            #    stream BEFORE update_state (top_rep image_idx = n_images before
            #    the update = len(class_imgs[cls]) at append time).
            pred_conf = F.softmax(gate_logits, dim=-1).squeeze(0)
            above_thresh = (pred_conf > conf_thresh).nonzero(as_tuple=True)[0]
            gated = [int(c.item()) for c in above_thresh]

            img_cpu = images.detach().cpu()
            all_views_cpu = [img_cpu]
            for _ in range(aug_copies):
                all_views_cpu.append(_augment_image(images).detach().cpu())
            for cls_idx in gated:
                class_imgs[cls_idx].append(all_views_cpu)
                states[cls_idx] = adapter.patch_level.update_state(
                    states[cls_idx], images, encoder, feat_norm,
                    target_class_idx=cls_idx,
                )

            # Snapshot the post-update state for deferred sample rendering.
            # update_state returns a fresh dict, so this shallow copy stays
            # frozen even as the live state evolves.
            if is_selected:
                sel = selected[i]
                sel["img_cpu"] = img_cpu
                sel["was_gated"] = (tgt in gated)
                sel["state_after"] = dict(states[tgt])

            if i % 60 == 0:
                running = 100.0 * sum(r["correct"] for r in replay_records) / len(replay_records)
                print("---- replay batch {i}: running acc {a:.2f}% ----".format(i=i, a=running))

    total = len(replay_records)
    final_acc = 100.0 * sum(r["correct"] for r in replay_records) / max(total, 1)
    per_class_acc = {}
    per_class_total = {}
    for c in range(C):
        tot = sum(1 for r in replay_records if r["target"] == c)
        cor = sum(1 for r in replay_records if r["target"] == c and r["correct"])
        per_class_total[c] = tot
        per_class_acc[c] = 100.0 * cor / tot if tot else 0.0

    # ── End-of-run prototype montages (FINAL per-class state) ────────────────
    print("\nRendering proto_final.png per class ...")
    for c in range(C):
        path = render_proto_final(
            c, classnames[c], states[c], class_imgs[c], per_class_acc[c],
            appearance_min_weight, out_dir / classnames[c],
        )
        print("  -> {}".format(path))

    # ── Deferred sample pages (post-loop; no RNG consumed mid-stream) ────────
    print("\nRendering sample pages ...")
    for i in sorted(selected):
        path = render_sample_page(
            i, selected[i], replay_records[i], classnames, encoder,
            adapter.fusion, out_dir,
        )
        print("  -> {}".format(path))

    # ── Per-class summary (write BEFORE fidelity gate so it survives exit) ──
    write_summary_json(out_dir, classnames, states, per_class_acc,
                       per_class_total, total, final_acc, args.seed)

    # ── Fidelity gate vs original records ───────────────────────────────────
    _header, orig_records = load_records(args.record)
    matched, mismatches = compare_fidelity(replay_records, orig_records, total)
    replay_check = {"matched": matched, "total": total, "mismatches": mismatches}
    with open(out_dir / "replay_check.json", "w") as f:
        json.dump(replay_check, f, indent=2)

    print("\n==============================================")
    print("REPLAY FIDELITY: {}/{} batches match".format(matched, total))
    print("==============================================")
    if matched < total:
        for m in mismatches[:5]:
            print("  MISMATCH batch {b} field {f}: expected={e!r} got={g!r}".format(
                b=m["batch_idx"], f=m["field"], e=m["expected"], g=m["got"]))
        if total >= 180:
            print("ERROR: replay fidelity gate FAILED — figures are NOT trustworthy.")
            sys.exit(1)
        print("WARNING: partial replay (max_batches={}) — mismatch beyond the "
              "replayed prefix is expected.".format(args.max_batches))

    # ── stdout cross-check vs records summary.json ──────────────────────────
    if total >= 180:
        record_summary_path = Path(args.record).parent / "summary.json"
        crosscheck_record_summary(record_summary_path, classnames, states,
                                  per_class_acc, final_acc, out_dir)

    print("\nDone.  Outputs written under: {}".format(out_dir.resolve()))


if __name__ == "__main__":
    main()
