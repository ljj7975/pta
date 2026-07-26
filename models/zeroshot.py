"""Zero-shot CLIP baseline — no test-time adaptation."""
import os
import torch

from tqdm import tqdm

from utils import get_clip_logits, cls_acc


class ZeroShotAdapter:
    """Pure zero-shot CLIP — no prototype updates, no fusion."""

    def __init__(self, cfg):
        self.cfg = cfg

    def run(self, loader, encoder, text_embeddings, dataset_name: str) -> float:
        os.makedirs("outputs", exist_ok=True)

        max_batches = os.environ.get("MAX_BATCHES")
        if max_batches is not None:
            max_batches = int(max_batches)

        with torch.no_grad():
            accuracies = []

            for i, (images, target) in enumerate(
                tqdm(loader, desc=f"[ZeroShot] {dataset_name}")
            ):
                if max_batches and i >= max_batches:
                    break

                _, clip_logits, _, _, _ = get_clip_logits(
                    images, encoder, text_embeddings
                )
                target = target.cuda()
                acc = cls_acc(clip_logits, target)
                accuracies.append(acc)

        final_acc = sum(accuracies) / len(accuracies)
        print(f"\n---- ZeroShot FINAL {final_acc:.2f}% ----\n")
        return final_acc


def build(cfg):
    return ZeroShotAdapter(cfg)
