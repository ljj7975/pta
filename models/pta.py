import os
import torch
from tqdm import tqdm

from models.base import BaseAdapter
from models.image_level import create as create_image_level
from models.fusion import WeightedFusion
from utils import get_clip_logits, cls_acc

# ─────────────────────────────────────────────────────────────────────────────
# PROTOTYPE-BASED TEST-TIME ADAPTATION (PTA)
# ─────────────────────────────────────────────────────────────────────────────
# High-level algorithm:
#   1. Start with zero-shot CLIP (text embeddings as initial prototypes)
#   2. For each test sample:
#      a. Get image features from CLIP encoder
#      b. Compute initial logits (zero-shot prediction)
#      c. For high-confidence classes, blend image feature into running prototype
#      d. Blend updated prototype with original text embedding
#      e. Compute refined logits and measure accuracy
#   3. Online updates use exponential moving average (controlled by T parameter)
#   4. Final prediction fuses zero-shot logits with refined prototype logits
# ─────────────────────────────────────────────────────────────────────────────


class PTAAdapter(BaseAdapter):
    """
    Prototype-Based Test-Time Adaptation (PTA).

    Implements the PTA method from "Prototype-Based Test-Time Adaptation of
    Vision-Language Models" (Huang et al., ICML 2026).

    Key idea: Maintain a per-class running prototype that is blended online from
    high-confidence image features. The final prediction fuses:
      1. Zero-shot CLIP logits (original model, no adaptation)
      2. Logits from refined text features (text + adapted prototype)

    This gives the model two "views" — the trusted zero-shot baseline and the
    test-time adapted view — which are combined for improved robustness.

    Reference: Huang et al., ICML 2026 — https://arxiv.org/abs/2604.21360
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
        cfg["image_level"] = image_level_cfg

        # ── Image-level prototype component ────────────────────────────────
        self.image_level = create_image_level(cfg)

        # ── Fusion component ───────────────────────────────────────────────
        # PTA defaults: tau_text=1.0, tau_image_proto=100.0, tau_patch_proto=0.0
        fusion_cfg = cfg.get("fusion", {})
        fusion_cfg.setdefault("tau_text", 1.0)
        fusion_cfg.setdefault("tau_image_proto", 100.0)
        fusion_cfg.setdefault("tau_patch_proto", 0.0)
        cfg["fusion"] = fusion_cfg
        self.fusion = WeightedFusion(cfg)

    def run(
        self,
        loader,
        encoder,
        text_embeddings,
        dataset_name: str,
    ) -> float:
        """
        Run PTA on test set. Per-dataset config is loaded from self.cfg.
        """
        os.makedirs("outputs", exist_ok=True)

        with torch.no_grad():
            accuracies = []                                     # Track per-sample accuracy

            # Initialize refined text features as original CLIP text embeddings
            # text_embeddings: [D, C] -> transpose to [C, D]
            refine_feature = text_embeddings.t()                # [C, D] fp16

            # Initialize prototype bank with zeros (filled by first samples via EMA)
            target_prototype = self.image_level.init_state(
                refine_feature
            )                                                   # [C, D]

            # Support early termination via MAX_BATCHES env var
            max_batches = os.environ.get("MAX_BATCHES")
            if max_batches is not None:
                max_batches = int(max_batches)

            # Main evaluation loop: process one test sample per iteration
            for i, (images, target) in enumerate(
                tqdm(loader, desc=f"[PTA] {dataset_name}")
            ):
                # Early termination check
                if max_batches is not None and i >= max_batches:
                    break

                # ── ZERO-SHOT PREDICTION ───────────────────────────────────
                image_features, clip_logits, _, _, _ = get_clip_logits(
                    images, encoder, text_embeddings
                )

                target = target.cuda()

                # ── ONLINE PROTOTYPE UPDATE ────────────────────────────────
                # PTAImageLevel applies softmax internally; pass raw logits.
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
                # WeightedFusion handles tau weights and tensor cloning.
                final_logits = self.fusion.forward(
                    clip_logits.clone(), image_proto_logits, None
                )

                # ── MEASURE ACCURACY ───────────────────────────────────────
                acc = cls_acc(final_logits, target)
                accuracies.append(acc)

                # Periodic logging (every 1000 samples)
                if i % 1000 == 0:
                    print(
                        f"---- PTA's test accuracy: "
                        f"{sum(accuracies)/len(accuracies):.2f}. ----"
                    )

        # ── FINAL RESULTS ──────────────────────────────────────────────────
        final_acc = sum(accuracies) / len(accuracies)
        print(f"---- PTA's test accuracy: {final_acc:.2f}. ----\n")

        # Append results to output file (append mode, multiple runs accumulate)
        label = os.environ.get("RESULT_LABEL", "PTA")
        with open("outputs/result.txt", "a") as f:
            f.write(
                f"{label}'s performance on {dataset_name}: Top1- {final_acc:.2f}.\n"
            )

        return final_acc


def build(cfg: dict) -> PTAAdapter:
    """
    Factory function: instantiate a PTAAdapter with the given config.

    Called by runner.py via dynamic import:
      adapter_module = __import__('models.pta', fromlist=['build'])
      adapter = adapter_module.build(cfg)
    """
    return PTAAdapter(cfg)
