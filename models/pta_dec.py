import os
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.base import BaseAdapter
from models.image_level import create as create_image_level
from models.fusion import WeightedFusion
from utils import get_clip_logits, cls_acc
from utils.records import write_record_header, write_record, write_summary

# ─────────────────────────────────────────────────────────────────────────────
# PROTOTYPE-BASED TEST-TIME ADAPTATION WITH DEC CERTAINTY REGULARIZER (PTA-DEC)
# ─────────────────────────────────────────────────────────────────────────────
# Extension of PTA with entropy + logit-norm aware confidence scoring.
#
# Key innovation: Detect unreliable pseudo-labels by combining:
#   1. Entropy (prediction uncertainty)
#   2. Logit norm (feature magnitude / confidence)
#   3. Adaptive temperature scaling (stronger smoothing for uncertain samples)
#
# This addresses the root cause of PTA's failure: 50% noisy pseudo-labels under
# distribution shift. Instead of structural upgrades (banks, Gaussians, gates),
# we improve confidence scoring to filter unreliable samples.
# ─────────────────────────────────────────────────────────────────────────────


class PTADECAdapter(BaseAdapter):
    """
    Prototype-Based Test-Time Adaptation with DEC Certainty Regularizer.

    Extends PTA with entropy + logit-norm aware confidence scoring to handle
    noisy pseudo-labels under distribution shift.

    Key idea: Maintain a per-class running prototype that is blended online from
    high-confidence image features. The final prediction fuses:
      1. Zero-shot CLIP logits (original model, no adaptation)
      2. Logits from refined text features (text + adapted prototype)

    The DEC certainty regularizer improves confidence scoring by:
      - Computing entropy from softmax probabilities
      - Computing logit norm as proxy for feature magnitude
      - Combining into adaptive temperature τ_i
      - Applying stronger smoothing to uncertain samples

    Reference: Huang et al., ICML 2026 — https://arxiv.org/abs/2604.21360
    DEC mechanism: Entropy + logit-norm aware confidence (TMLR 2025)
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        # ── Backward-compatible nested config ──────────────────────────────
        # Flat configs (alpha, T at root level) are still used by some
        # callers.  Propagate them into the nested "image_level" sub-dict
        # so the PTAImageLevel component can find them.
        image_level_cfg = cfg.get("image_level", {})
        if "alpha" not in image_level_cfg and "alpha" in cfg:
            image_level_cfg["alpha"] = cfg["alpha"]
        if "T" not in image_level_cfg and "T" in cfg:
            image_level_cfg["T"] = cfg["T"]
        
        # Enable DEC by default for this adapter
        if "use_dec" not in image_level_cfg:
            image_level_cfg["use_dec"] = True
        
        # Set DEC hyperparameters from config or use defaults
        if "entropy_baseline" not in image_level_cfg:
            image_level_cfg["entropy_baseline"] = cfg.get("entropy_baseline", 0.5)
        if "temp_min" not in image_level_cfg:
            image_level_cfg["temp_min"] = cfg.get("temp_min", 0.5)
        if "temp_max" not in image_level_cfg:
            image_level_cfg["temp_max"] = cfg.get("temp_max", 1.0)
        if "logit_norm_weight" not in image_level_cfg:
            image_level_cfg["logit_norm_weight"] = cfg.get("logit_norm_weight", 0.1)
        
        cfg["image_level"] = image_level_cfg

        # ── Image-level prototype component (DEC-enhanced) ────────────────
        # Use pta_image_dec instead of pta_image
        self.image_level = self._create_image_level_dec(cfg)

        # ── Fusion component ───────────────────────────────────────────────
        # PTA defaults: tau_text=1.0, tau_image_proto=100.0, tau_patch_proto=0.0
        fusion_cfg = cfg.get("fusion", {})
        fusion_cfg.setdefault("tau_text", 1.0)
        fusion_cfg.setdefault("tau_image_proto", 100.0)
        fusion_cfg.setdefault("tau_patch_proto", 0.0)
        cfg["fusion"] = fusion_cfg
        self.fusion = WeightedFusion(cfg)

    def _create_image_level_dec(self, cfg):
        """Create DEC-enhanced image level adapter."""
        from models.image_level.pta_image_dec import PTAImageLevelDEC
        return PTAImageLevelDEC(cfg)

    def run(
        self,
        loader,
        encoder,
        text_embeddings,
        dataset_name: str,
    ) -> float:
        """
        Run PTA-DEC on test set. Per-dataset config is loaded from self.cfg.
        """
        os.makedirs("outputs", exist_ok=True)

        # ── ENV-GATED PER-SAMPLE RECORDING (behavior-neutral) ─────────────
        record_dir = os.environ.get("RECORD_DIR", "").strip()
        if record_dir:
            write_record_header(
                method=os.environ.get("RESULT_LABEL", "PTA-DEC"),
                dataset=dataset_name,
                seed=int(os.environ.get("SEED", 1)),
                C=int(text_embeddings.shape[1]),
                classnames=[],
                resolved_config=self.cfg,
            )

        with torch.no_grad():
            accuracies = []                                     # Track per-sample accuracy

            # Initialize refined text features as original CLIP text embeddings
            # text_embeddings: [D, C] -> transpose to [C, D]
            refine_feature = text_embeddings.t()                # [C, D] fp16

            # Initialize prototype bank with zeros (filled by first samples via EMA)
            target_prototype = self.image_level.init_state(
                refine_feature
            )                                                   # [C, D]

            # Per-class accuracy accumulators for the summary's per_class dict.
            num_classes = text_embeddings.shape[1]
            cls_total = [0] * num_classes
            cls_correct = [0] * num_classes

            # Support early termination via MAX_BATCHES env var
            max_batches = os.environ.get("MAX_BATCHES")
            if max_batches is not None:
                max_batches = int(max_batches)

            # Main evaluation loop: process one test sample per iteration
            for i, (images, target) in enumerate(
                tqdm(loader, desc=f"[PTA-DEC] {dataset_name}")
            ):
                # Early termination check
                if max_batches is not None and i >= max_batches:
                    break

                # ── ZERO-SHOT PREDICTION ───────────────────────────────────
                image_features, clip_logits, _, _, _ = get_clip_logits(
                    images, encoder, text_embeddings
                )

                target = target.cuda()

                # ── ONLINE PROTOTYPE UPDATE (DEC-ENHANCED) ─────────────────
                # PTAImageLevelDEC applies entropy + logit-norm aware confidence
                refine_feature, target_prototype = (
                    self.image_level.update_prototypes(
                        image_features,
                        clip_logits,
                        refine_feature,
                        target_prototype,
                    )
                )

                # ── FUSED PREDICTION ───────────────────────────────────────
                # Compute image-level prototype logits
                image_proto_logits = self.image_level.compute_logits(
                    image_features, refine_feature
                )

                # Fuse zero-shot logits with prototype logits
                final_logits = self.fusion.forward(
                    clip_logits.clone(), image_proto_logits, None
                )

                # ── MEASURE ACCURACY ───────────────────────────────────────
                acc = cls_acc(final_logits, target)
                accuracies.append(acc)

                # Per-class accuracy accumulation
                cls_total[int(target.item())] += 1
                if acc:
                    cls_correct[int(target.item())] += 1

                # ── RECORD PER-SAMPLE (env-gated, behavior-neutral) ───────
                if record_dir:
                    write_record(
                        batch_idx=i,
                        target=int(target.item()),
                        pred=int(final_logits.argmax(dim=-1).item()),
                        correct=bool(acc),
                        conf=float(F.softmax(final_logits, dim=-1).max().item()),
                        quality_gate=None,
                        gate_mode=None,
                        proto_alpha=None,
                        logits={
                            "clip": clip_logits.squeeze(0).float().cpu().tolist(),
                            "image_proto": image_proto_logits.squeeze(0).float().cpu().tolist(),
                            "patch_proto": None,
                            "final": final_logits.squeeze(0).float().cpu().tolist(),
                        },
                        proto_stats={"true": None, "pred": None},
                    )

                # Periodic logging (every 1000 samples)
                if i % 1000 == 0:
                    print(
                        f"---- PTA-DEC's test accuracy: "
                        f"{sum(accuracies)/len(accuracies):.2f}. ----"
                    )

        # ── FINAL RESULTS ──────────────────────────────────────────────────
        final_acc = sum(accuracies) / len(accuracies)
        print(f"---- PTA-DEC's test accuracy: {final_acc:.2f}. ----\n")

        # ── SUMMARY (env-gated, behavior-neutral) ──────────────────────────
        if record_dir:
            per_class = {}
            for c in range(num_classes):
                total_c = cls_total[c]
                correct_c = cls_correct[c]
                per_class["class_{}".format(c)] = {
                    "total": total_c,
                    "correct": correct_c,
                    "acc": (100.0 * correct_c / total_c) if total_c else 0.0,
                }
            write_summary(
                method=os.environ.get("RESULT_LABEL", "PTA-DEC"),
                dataset=dataset_name,
                seed=int(os.environ.get("SEED", 1)),
                total=len(accuracies),
                acc=final_acc,
                per_class=per_class,
            )

        # Append results to output file (append mode, multiple runs accumulate)
        label = os.environ.get("RESULT_LABEL", "PTA-DEC")
        result_file = os.environ.get("RESULT_FILE", "outputs/result.txt")
        with open(result_file, "a") as f:
            f.write(
                f"{label}'s performance on {dataset_name}: Top1- {final_acc:.2f}.\n"
            )

        return final_acc


def build(cfg: dict) -> PTADECAdapter:
    """
    Factory function: instantiate a PTADECAdapter with the given config.

    Called by runner.py via dynamic import:
      adapter_module = __import__('models.pta_dec', fromlist=['build'])
      adapter = adapter_module.build(cfg)
    """
    return PTADECAdapter(cfg)
