import os
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.base import BaseAdapter
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


def update_text_features(
    image_feature: torch.Tensor,
    probs: torch.Tensor,
    text_features: torch.Tensor,
    target_prototype: torch.Tensor,
    alpha: float = 0.01,
    T: float = 20.0,
):
    """
    Online prototype update step.

    For each class whose soft-max probability exceeds 0.1, blend the current
    image feature into the running prototype, then form the refined text
    feature as a convex combination of the original text embedding and the
    prototype.

    Args:
        image_feature:   (1, D) L2-normalised image embedding from CLIP vision encoder.
        probs:           (1, C) soft-max distribution over classes (from zero-shot logits).
        text_features:   (C, D) original CLIP text embeddings (class names via text encoder).
        target_prototype:(C, D) running prototype bank (mutated in-place with EMA).
        alpha:           Weight on original text features (0.0 = pure prototype, 1.0 = no update).
        T:               Temperature controlling prototype update rate.
                         Higher T → more conservative updates (exponential decay).

    Returns:
        refined_text:    (C, D) L2-normalised updated text features (text + prototype blend).
        target_prototype:(C, D) updated prototype bank (after EMA step).
    """
    # Extract soft probabilities [C] from batch dim
    w = probs.squeeze(0)                          # [C] — class confidence from zero-shot
    
    # Compute update weights via exponential decay: w_new = 1 - exp(-w / T)
    # Only apply to high-confidence classes (w >= 0.1)
    w_new = torch.zeros_like(w)                   # [C] — init zero
    mask = w >= 1e-1                              # [C] bool — which classes to update
    w_new[mask] = 1 - torch.exp(-w[mask] / T)     # [C] — update weight for confident classes
    w_new = w_new.unsqueeze(1)                    # [C, 1] — reshape for broadcast

    # EMA update: blend old prototype with new image feature
    # target_prototype[c] = (1 - w_new[c]) * old_proto[c] + w_new[c] * image_feature
    # High w_new = strong update; low w_new = conservative update
    target_prototype[mask] = (
        (1 - w_new[mask]) * target_prototype[mask]
        + w_new[mask] * image_feature.squeeze(0)
    )

    # Form refined text features as a blend of original CLIP text + updated prototype
    # alpha controls the balance:
    #   alpha=1.0 → pure original text (no adaptation)
    #   alpha=0.0 → pure prototype (full adaptation)
    refined_text = alpha * text_features + (1 - alpha) * target_prototype  # [C, D]
    
    # L2-normalize so cosine similarity = dot product
    refined_text = refined_text / refined_text.norm(dim=-1, keepdim=True)  # [C, D]

    return refined_text, target_prototype


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

    def run(
        self,
        loader,
        clip_model,
        clip_weights,
        dataset_name: str,
    ) -> float:
        """
        Run PTA on test set. Per-dataset config is loaded from self.cfg.
        """
        # Load hyperparameters from per-dataset YAML config
        alpha = self.cfg.get("alpha", 0.01)
        T = float(self.cfg.get("T", 20.0))

        os.makedirs("outputs", exist_ok=True)

        with torch.no_grad():
            accuracies = []                                           # Track per-sample accuracy
            
            # Initialize refined text features as original CLIP text embeddings
            # clip_weights: [D, C] → transpose to [C, D]
            refine_feature = clip_weights.t()                         # [C, D] — text embeddings
            
            # Initialize prototype bank with zeros (will be filled by first samples)
            # Will accumulate via EMA during the test loop
            target_prototype = torch.zeros_like(refine_feature).cuda()  # [C, D]

            # Main evaluation loop: process one test sample per iteration
            for i, (images, target) in enumerate(
                tqdm(loader, desc=f"[PTA] {dataset_name}")
            ):
                # ── ZERO-SHOT PREDICTION ───────────────────────────────────
                # Get image features from CLIP vision encoder + zero-shot logits
                image_features, clip_logits, _, _, _ = get_clip_logits(
                    images, clip_model, clip_weights
                )
                # image_features: [1, D] — L2-normalized CLIP image embedding
                # clip_logits: [1, C] — zero-shot logits (image @ text^T)
                
                target = target.cuda()  # Ground truth label

                # ── ONLINE PROTOTYPE UPDATE ────────────────────────────────
                # Convert logits to class probabilities (soft labels)
                soft_logits = F.softmax(clip_logits, dim=-1)  # [1, C] — soft probabilities
                
                # Update the running prototype and refined text features
                # High-confidence classes get their prototypes updated via EMA
                refine_feature, target_prototype = update_text_features(
                    image_features,
                    soft_logits.half(),
                    refine_feature,
                    target_prototype,
                    alpha=alpha,      # Balance: original text vs. prototype
                    T=T,              # Decay temperature for update rate
                )
                # refine_feature: [C, D] — updated text features (text + prototype blend)
                # target_prototype: [C, D] — updated prototype bank (after EMA)

                # ── FUSED PREDICTION ───────────────────────────────────────
                # Combine zero-shot logits with logits from refined text features
                final_logits = clip_logits.clone()  # [1, C] — start with zero-shot
                
                # Add contribution from refined features (with large scaling factor 100.0)
                # This is cosine similarity: image @ refined_text^T, scaled up
                final_logits += 100.0 * image_features.half() @ refine_feature.half().T  # [1, C]
                # Scaling factor is large (100.0) to ensure refined branch is meaningful

                # ── MEASURE ACCURACY ───────────────────────────────────────
                acc = cls_acc(final_logits, target)  # Compute top-1 accuracy
                accuracies.append(acc)

                # Periodic logging (every 1000 samples)
                if i % 1000 == 0:
                    print(
                        f"---- PTA's test accuracy: "
                        f"{sum(accuracies)/len(accuracies):.2f}. ----"
                    )

        # ── FINAL RESULTS ──────────────────────────────────────────────
        final_acc = sum(accuracies) / len(accuracies)  # Average accuracy over all test samples
        print(f"---- PTA's test accuracy: {final_acc:.2f}. ----\n")

        # Append results to output file (append mode, so multiple runs accumulate)
        with open("outputs/result.txt", "a") as f:
            f.write(
                f"PTA's performance on {dataset_name}: Top1- {final_acc:.2f}.\n"
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
