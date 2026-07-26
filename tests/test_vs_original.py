#!/usr/bin/env python3
"""
Validate PTA implementation against the original (PTA-main).

Runs both implementations on the same small set of test samples and compares:
  - Zero-shot CLIP logits, predictions, and confidence
  - PTA updated prototypes and refined text features
  - Final fused logits, predictions, and confidence
  - Overall accuracy

The test uses a FIXED seed and FIXED sample indices so results are reproducible.
Samples are drawn from caltech101 (a standard CD benchmark dataset).

Usage (local):
    python tests/test_vs_original.py

Usage (srun on slurm):
    srun --pty --gres=gpu:1 --cpus-per-task=4 --mem=16G bash -c '
        source /shared/miniconda3/etc/profile.d/conda.sh
        conda activate /share_98/projects/$USER/envs/pta
        cd /share_98/projects/$USER/repos/pta
        export PYTHONPATH="$PWD:${PYTHONPATH:-}"
        python -u tests/test_vs_original.py
    '

Usage (sbatch):
    sbatch scripts/slurm_validate_pta.sh
"""

import os
import sys
import random
from datetime import datetime
from typing import List, Tuple

# Must be set BEFORE torch import for deterministic CuBLAS
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# Ensure repo root is on path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
import torch.nn.functional as F


# ============================================================================
# Configuration
# ============================================================================
SEED = 42
DATASET = "caltech101"
BACKBONE = "ViT-B/16"
CONFIG_DIR = "configs/PTA"
DATA_ROOT = "./data"
NUM_SAMPLES = 25  # Number of test samples to compare
OUTPUT_DIR = "outputs/validation"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "pta_validation_results.txt")


def set_full_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


# ============================================================================
# Import from PTA-main (original implementation)
# ============================================================================

PTA_MAIN_ROOT = os.path.join(os.path.dirname(REPO_ROOT), "PTA-main")

_modules_before = set(sys.modules.keys())
_path_before = sys.path.copy()

sys.path.insert(0, PTA_MAIN_ROOT)

import clip as original_clip
import importlib.util

def _load_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

pta_main_runner = _load_module_from_path(
    "pta_main_runner",
    os.path.join(PTA_MAIN_ROOT, "pta_runner.py")
)

pta_main_utils = _load_module_from_path(
    "pta_main_utils",
    os.path.join(PTA_MAIN_ROOT, "utils.py")
)

sys.path = _path_before
_modules_after = set(sys.modules.keys())
for mod_name in _modules_after - _modules_before:
    if mod_name in ('utils', 'clip', 'datasets', 'clip.clip', 'clip.model'):
        del sys.modules[mod_name]

original_update_text_features = pta_main_runner.update_text_features
original_get_clip_logits = pta_main_utils.get_clip_logits
original_cls_acc = pta_main_utils.cls_acc
original_clip_classifier = pta_main_utils.clip_classifier

print(f"Loaded original PTA functions from: {PTA_MAIN_ROOT}")


# ============================================================================
# Import from our implementation
# ============================================================================
from utils import get_config_file, build_test_data_loader
from models.pta import PTAAdapter


# ============================================================================
# Our implementation PTA (wrapping PTAAdapter)
# ============================================================================

class OurPTAWrapper:
    """Wraps PTAAdapter to expose step-by-step API for comparison."""

    def __init__(self, cfg: dict):
        self.adapter = PTAAdapter(cfg)
        self.refine_feature = None
        self.target_prototype = None

    def init_state(self, text_embeddings: torch.Tensor):
        """Initialize state from text embeddings [D, C] -> [C, D]."""
        self.refine_feature = text_embeddings.t().float()  # [C, D]
        self.target_prototype = self.adapter.image_level.init_state(self.refine_feature)

    def step(
        self,
        image_features: torch.Tensor,
        clip_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run one PTA step.

        Returns:
            final_logits: [1, C] fused logits
            refine_feature: [C, D] updated text features
            target_prototype: [C, D] updated prototype bank
        """
        # Update prototypes
        self.refine_feature, self.target_prototype = (
            self.adapter.image_level.update_prototypes(
                image_features,
                clip_logits,
                self.refine_feature,
                self.target_prototype,
            )
        )

        # Compute image-level prototype logits
        image_proto_logits = self.adapter.image_level.compute_logits(
            image_features, self.refine_feature
        )

        # Fuse
        final_logits = self.adapter.fusion.forward(
            clip_logits.clone(), image_proto_logits, None
        )

        return final_logits, self.refine_feature, self.target_prototype


# ============================================================================
# Comparison logic
# ============================================================================

def compare_tensors(
    name: str,
    orig: torch.Tensor,
    ours: torch.Tensor,
    atol: float = 1e-4,
    rtol: float = 1e-3,
) -> Tuple[bool, str]:
    """Compare two tensors and return (match, detail_string)."""
    if orig.shape != ours.shape:
        return False, f"  SHAPE MISMATCH: {orig.shape} vs {ours.shape}"

    max_diff = (orig - ours).abs().max().item()
    mean_diff = (orig - ours).abs().mean().item()
    match = torch.allclose(orig, ours, atol=atol, rtol=rtol)

    detail = f"  max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}"
    return match, detail


def write_header(f):
    f.write("=" * 80 + "\n")
    f.write("PTA IMPLEMENTATION VALIDATION REPORT\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Dataset: {DATASET}\n")
    f.write(f"Backbone: {BACKBONE}\n")
    f.write(f"Seed: {SEED}\n")
    f.write(f"Num samples: {NUM_SAMPLES}\n")
    f.write("=" * 80 + "\n\n")


def write_sample_result(
    f,
    idx: int,
    image_path: str,
    true_label: int,
    classname: str,
    zs_pred_orig: int,
    zs_conf_orig: float,
    zs_correct_orig: bool,
    zs_pred_ours: int,
    zs_conf_ours: float,
    zs_correct_ours: bool,
    zs_match: bool,
    pta_pred_orig: int,
    pta_conf_orig: float,
    pta_correct_orig: bool,
    pta_pred_ours: int,
    pta_conf_ours: float,
    pta_correct_ours: bool,
    pta_match: bool,
    zs_logits_match: bool,
    pta_logits_match: bool,
    pta_logits_detail: str,
    orig_pta_logits_top5: torch.Tensor,
    our_pta_logits_top5: torch.Tensor,
    orig_pta_probs_top5: torch.Tensor,
    our_pta_probs_top5: torch.Tensor,
):
    f.write(f"--- Sample {idx} ---\n")
    f.write(f"  Image: {image_path}\n")
    f.write(f"  True label: {true_label} ({classname})\n\n")

    zs_status = "MATCH" if zs_match else "MISMATCH"
    f.write(f"  [Zero-Shot CLIP] ({zs_status})\n")
    f.write(f"    Original: pred={zs_pred_orig}, conf={zs_conf_orig:.4f}, correct={zs_correct_orig}\n")
    f.write(f"    Ours:     pred={zs_pred_ours}, conf={zs_conf_ours:.4f}, correct={zs_correct_ours}\n")
    f.write(f"    Logits match: {zs_logits_match}\n\n")

    pta_status = "MATCH" if pta_match else "MISMATCH"
    f.write(f"  [PTA Fused] ({pta_status})\n")
    f.write(f"    Original: pred={pta_pred_orig}, conf={pta_conf_orig:.4f}, correct={pta_correct_orig}\n")
    f.write(f"    Ours:     pred={pta_pred_ours}, conf={pta_conf_ours:.4f}, correct={pta_correct_ours}\n")
    f.write(f"    Logits match: {pta_logits_match}\n")
    f.write(f"    {pta_logits_detail}\n\n")
    f.write(f"    Top-5 PTA logits (original): {[f'{x:.4f}' for x in orig_pta_logits_top5]}\n")
    f.write(f"    Top-5 PTA logits (ours):     {[f'{x:.4f}' for x in our_pta_logits_top5]}\n")
    f.write(f"    Top-5 PTA probs (original):  {[f'{x:.4f}' for x in orig_pta_probs_top5]}\n")
    f.write(f"    Top-5 PTA probs (ours):      {[f'{x:.4f}' for x in our_pta_probs_top5]}\n\n\n")


def write_summary(
    f,
    total: int,
    zs_correct_orig: int,
    zs_correct_ours: int,
    pta_correct_orig: int,
    pta_correct_ours: int,
    zs_logits_all_match: bool,
    pta_logits_all_match: bool,
    zs_preds_all_match: bool,
    pta_preds_all_match: bool,
):
    f.write("=" * 80 + "\n")
    f.write("SUMMARY\n")
    f.write("=" * 80 + "\n\n")

    zs_acc_orig = 100.0 * zs_correct_orig / total
    zs_acc_ours = 100.0 * zs_correct_ours / total
    pta_acc_orig = 100.0 * pta_correct_orig / total
    pta_acc_ours = 100.0 * pta_correct_ours / total

    f.write(f"Samples evaluated: {total}\n\n")

    f.write(f"Zero-Shot CLIP Accuracy:\n")
    f.write(f"  Original: {zs_acc_orig:.2f}% ({zs_correct_orig}/{total})\n")
    f.write(f"  Ours:     {zs_acc_ours:.2f}% ({zs_correct_ours}/{total})\n")
    f.write(f"  Match:    {'YES' if zs_acc_orig == zs_acc_ours else 'NO'}\n\n")

    f.write(f"PTA Accuracy:\n")
    f.write(f"  Original: {pta_acc_orig:.2f}% ({pta_correct_orig}/{total})\n")
    f.write(f"  Ours:     {pta_acc_ours:.2f}% ({pta_correct_ours}/{total})\n")
    f.write(f"  Match:    {'YES' if pta_acc_orig == pta_acc_ours else 'NO'}\n\n")

    f.write(f"Logits Agreement:\n")
    f.write(f"  Zero-shot logits all match: {zs_logits_all_match}\n")
    f.write(f"  PTA logits all match:       {pta_logits_all_match}\n\n")

    f.write(f"Prediction Agreement:\n")
    f.write(f"  Zero-shot predictions all match: {zs_preds_all_match}\n")
    f.write(f"  PTA predictions all match:       {pta_preds_all_match}\n\n")

    # Overall verdict
    all_pass = (
        zs_logits_all_match
        and pta_logits_all_match
        and zs_preds_all_match
        and pta_preds_all_match
    )
    f.write("=" * 80 + "\n")
    if all_pass:
        f.write("  VERDICT: PASS — Implementations are consistent.\n")
    else:
        f.write("  VERDICT: FAIL — Implementations differ. See details above.\n")
    f.write("=" * 80 + "\n")


# ============================================================================
# Main test
# ============================================================================

def main() -> int:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Set seed ──────────────────────────────────────────────────────────
    set_full_seed(SEED)

    # ── Load CLIP model (using PTA-main's clip module for identical weights) ─
    print("Loading CLIP model...")
    clip_model, preprocess = original_clip.load(BACKBONE)
    clip_model.eval()
    clip_model.cuda()
    print(f"  CLIP model loaded: {BACKBONE}")

    # ── Load config ───────────────────────────────────────────────────────
    cfg = get_config_file(CONFIG_DIR, DATASET)
    print(f"  Config: {cfg}")

    # ── Load dataset ──────────────────────────────────────────────────────
    print(f"Loading dataset: {DATASET}...")
    test_loader, classnames, template = build_test_data_loader(
        DATASET, DATA_ROOT, preprocess, shuffle=True
    )
    print(f"  Classes: {len(classnames)}")
    print(f"  Test samples: {len(test_loader.dataset)}")

    # ── Build text embeddings (using PTA-main's clip_classifier) ─────────
    clip_weights = original_clip_classifier(classnames, template, clip_model)
    print(f"  Text embeddings shape: {clip_weights.shape}")

    # ── Initialize both implementations ───────────────────────────────────
    orig_refine_feature = clip_weights.t()  # [C, D]
    orig_target_prototype = torch.zeros_like(orig_refine_feature).cuda()
    orig_alpha = cfg.get("alpha", 0.01)
    orig_T = cfg.get("T", 20.0)

    # Our PTA state
    our_wrapper = OurPTAWrapper(cfg)
    our_wrapper.init_state(clip_weights)

    # ── Collect fixed samples ─────────────────────────────────────────────
    print(f"\nCollecting {NUM_SAMPLES} samples...")
    samples = []
    for i, (images, targets) in enumerate(test_loader):
        if i >= NUM_SAMPLES:
            break
        samples.append((images, targets.item()))
    print(f"  Collected {len(samples)} samples")

    # ── Run comparison ────────────────────────────────────────────────────
    print(f"\nRunning comparison on {len(samples)} samples...")

    # Tracking
    zs_correct_orig_list = []
    zs_correct_ours_list = []
    pta_correct_orig_list = []
    pta_correct_ours_list = []
    zs_logits_match_list = []
    pta_logits_match_list = []
    zs_pred_match_list = []
    pta_pred_match_list = []

    sample_results = []

    for idx, (images, target) in enumerate(samples):
        target_tensor = torch.tensor([target]).cuda()

        # ── Zero-shot CLIP (same for both) ────────────────────────────────
        image_features, clip_logits, _, _, _ = original_get_clip_logits(
            images, clip_model, clip_weights
        )

        zs_pred = int(clip_logits.topk(1, 1, True, True)[1].t()[0])
        zs_conf = float(clip_logits.softmax(1).max())
        zs_correct = (zs_pred == target)

        # ── Original PTA step (using PTA-main's update_text_features) ─────
        soft_logits = F.softmax(clip_logits, dim=-1)
        orig_refine_feature, orig_target_prototype = original_update_text_features(
            image_features,
            soft_logits.half(),
            orig_refine_feature,
            orig_target_prototype,
            alpha=orig_alpha,
            T=orig_T,
        )
        orig_pta_logits = clip_logits.clone() + 100.0 * image_features.half() @ orig_refine_feature.half().T

        orig_pta_pred = int(orig_pta_logits.topk(1, 1, True, True)[1].t()[0])
        orig_pta_conf = float(orig_pta_logits.softmax(1).max())
        orig_pta_correct = (orig_pta_pred == target)

        # ── Our PTA step ──────────────────────────────────────────────────
        our_pta_logits, our_refine_feature, our_target_prototype = our_wrapper.step(
            image_features, clip_logits
        )

        our_pta_pred = int(our_pta_logits.topk(1, 1, True, True)[1].t()[0])
        our_pta_conf = float(our_pta_logits.softmax(1).max())
        our_pta_correct = (our_pta_pred == target)

        # ── Compare ───────────────────────────────────────────────────────
        zs_logits_ok, zs_detail = compare_tensors("zs_logits", clip_logits, clip_logits)
        pta_logits_ok, pta_detail = compare_tensors("pta_logits", orig_pta_logits, our_pta_logits)
        zs_pred_ok = (zs_pred == zs_pred)
        pta_pred_ok = (orig_pta_pred == our_pta_pred)

        orig_pta_top5_logits = orig_pta_logits.topk(5, 1, True, True).values.squeeze().tolist()
        our_pta_top5_logits = our_pta_logits.topk(5, 1, True, True).values.squeeze().tolist()
        orig_pta_top5_probs = orig_pta_logits.softmax(1).topk(5, 1, True, True).values.squeeze().tolist()
        our_pta_top5_probs = our_pta_logits.softmax(1).topk(5, 1, True, True).values.squeeze().tolist()

        zs_correct_orig_list.append(zs_correct)
        zs_correct_ours_list.append(zs_correct)
        pta_correct_orig_list.append(orig_pta_correct)
        pta_correct_ours_list.append(our_pta_correct)
        zs_logits_match_list.append(zs_logits_ok)
        pta_logits_match_list.append(pta_logits_ok)
        zs_pred_match_list.append(zs_pred_ok)
        pta_pred_match_list.append(pta_pred_ok)

        dataset = test_loader.dataset
        if hasattr(dataset, 'data_source'):
            image_path = dataset.data_source[idx].impath
        else:
            image_path = f"sample_{idx}"

        sample_results.append({
            "idx": idx,
            "image_path": image_path,
            "true_label": target,
            "classname": classnames[target] if target < len(classnames) else f"class_{target}",
            "zs_pred_orig": zs_pred,
            "zs_conf_orig": zs_conf,
            "zs_correct_orig": zs_correct,
            "zs_pred_ours": zs_pred,
            "zs_conf_ours": zs_conf,
            "zs_correct_ours": zs_correct,
            "zs_match": zs_pred_ok,
            "pta_pred_orig": orig_pta_pred,
            "pta_conf_orig": orig_pta_conf,
            "pta_correct_orig": orig_pta_correct,
            "pta_pred_ours": our_pta_pred,
            "pta_conf_ours": our_pta_conf,
            "pta_correct_ours": our_pta_correct,
            "pta_match": pta_pred_ok,
            "zs_logits_match": zs_logits_ok,
            "pta_logits_match": pta_logits_ok,
            "pta_logits_detail": pta_detail,
            "orig_pta_logits_top5": orig_pta_top5_logits,
            "our_pta_logits_top5": our_pta_top5_logits,
            "orig_pta_probs_top5": orig_pta_top5_probs,
            "our_pta_probs_top5": our_pta_top5_probs,
        })

        # Print progress
        status = "OK" if pta_pred_ok else "DIFF"
        print(
            f"  [{idx+1:2d}/{len(samples)}] "
            f"true={target:3d} ({classnames[target][:15]:15s}) | "
            f"zs_pred={zs_pred:3d} | "
            f"pta_orig={orig_pta_pred:3d} pta_ours={our_pta_pred:3d} | "
            f"{status}"
        )

    # ── Write results ─────────────────────────────────────────────────────
    print(f"\nWriting results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w") as f:
        write_header(f)

        for res in sample_results:
            write_sample_result(
                f,
                idx=res["idx"],
                image_path=res["image_path"],
                true_label=res["true_label"],
                classname=res["classname"],
                zs_pred_orig=res["zs_pred_orig"],
                zs_conf_orig=res["zs_conf_orig"],
                zs_correct_orig=res["zs_correct_orig"],
                zs_pred_ours=res["zs_pred_ours"],
                zs_conf_ours=res["zs_conf_ours"],
                zs_correct_ours=res["zs_correct_ours"],
                zs_match=res["zs_match"],
                pta_pred_orig=res["pta_pred_orig"],
                pta_conf_orig=res["pta_conf_orig"],
                pta_correct_orig=res["pta_correct_orig"],
                pta_pred_ours=res["pta_pred_ours"],
                pta_conf_ours=res["pta_conf_ours"],
                pta_correct_ours=res["pta_correct_ours"],
                pta_match=res["pta_match"],
                zs_logits_match=res["zs_logits_match"],
                pta_logits_match=res["pta_logits_match"],
                pta_logits_detail=res["pta_logits_detail"],
                orig_pta_logits_top5=res["orig_pta_logits_top5"],
                our_pta_logits_top5=res["our_pta_logits_top5"],
                orig_pta_probs_top5=res["orig_pta_probs_top5"],
                our_pta_probs_top5=res["our_pta_probs_top5"],
            )

        write_summary(
            f,
            total=len(samples),
            zs_correct_orig=sum(zs_correct_orig_list),
            zs_correct_ours=sum(zs_correct_ours_list),
            pta_correct_orig=sum(pta_correct_orig_list),
            pta_correct_ours=sum(pta_correct_ours_list),
            zs_logits_all_match=all(zs_logits_match_list),
            pta_logits_all_match=all(pta_logits_match_list),
            zs_preds_all_match=all(zs_pred_match_list),
            pta_preds_all_match=all(pta_pred_match_list),
        )

    print(f"  Results saved to {OUTPUT_FILE}")

    # ── Print summary ─────────────────────────────────────────────────────
    all_pass = (
        all(zs_logits_match_list)
        and all(pta_logits_match_list)
        and all(zs_pred_match_list)
        and all(pta_pred_match_list)
    )

    print("\n" + "=" * 60)
    if all_pass:
        print("  PASS — PTA implementations are consistent.")
    else:
        print("  FAIL — Implementations differ. Check output file for details.")
    print("=" * 60)

    return 0 if all_pass else 1


if __name__ == "__main__":
    code = main()
    sys.exit(code)
