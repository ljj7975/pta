"""Zero-shot CLIP baseline — no test-time adaptation."""
import os
import torch
import torch.nn.functional as F

from tqdm import tqdm

from utils import get_clip_logits, cls_acc
from utils.records import write_record_header, write_record, write_summary


class ZeroShotAdapter:
    """Pure zero-shot CLIP — no prototype updates, no fusion."""

    def __init__(self, cfg):
        self.cfg = cfg

    def run(self, loader, encoder, text_embeddings, dataset_name: str) -> float:
        os.makedirs("outputs", exist_ok=True)

        # Env-gated per-sample recording (see utils.records, T1). No-op unless
        # RECORD_DIR is set; recording never perturbs the inference path.
        record_dir = os.environ.get("RECORD_DIR", "").strip()
        if record_dir:
            os.makedirs(record_dir, exist_ok=True)
            write_record_header(
                method=os.environ.get("RESULT_LABEL", "ZeroShot"),
                dataset=dataset_name,
                seed=int(os.environ.get("SEED", "1")),
                C=text_embeddings.shape[1],
                classnames=[],
                resolved_config=self.cfg,
            )

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
                if record_dir:
                    write_record(
                        batch_idx=i,
                        target=int(target.item()),
                        pred=int(clip_logits.argmax(dim=-1).item()),
                        correct=bool(acc),
                        conf=float(F.softmax(clip_logits, dim=-1).max().item()),
                        quality_gate=None,
                        gate_mode=None,
                        proto_alpha=None,
                        logits={
                            "clip": clip_logits.squeeze(0).float().cpu().tolist(),
                            "image_proto": None,
                            "patch_proto": None,
                            "final": clip_logits.squeeze(0).float().cpu().tolist(),
                        },
                        proto_stats={"true": None, "pred": None},
                    )
                accuracies.append(acc)

        final_acc = sum(accuracies) / len(accuracies)
        print(f"\n---- ZeroShot FINAL {final_acc:.2f}% ----\n")

        if record_dir:
            write_summary(
                method=os.environ.get("RESULT_LABEL", "ZeroShot"),
                dataset=dataset_name,
                seed=int(os.environ.get("SEED", "1")),
                total=len(accuracies),
                acc=final_acc,
                per_class={},
            )

        label = os.environ.get("RESULT_LABEL", "ZeroShot")
        result_file = os.environ.get("RESULT_FILE", "outputs/result.txt")
        with open(result_file, "a") as f:
            f.write(
                f"{label}'s performance on {dataset_name}: "
                f"Top1- {final_acc:.2f}.\n"
            )

        return final_acc


def build(cfg):
    return ZeroShotAdapter(cfg)
