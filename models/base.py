from abc import ABC, abstractmethod
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# ARCHITECTURE OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
# This module defines a plugin interface for test-time adaptation (TTA) methods.
# Runner.py orchestrates:
#   1. Load a CLIP model (zero-shot)
#   2. For each dataset, load per-dataset config
#   3. Instantiate an adapter via models.<method>.build(cfg)
#   4. Call adapter.run(...) to evaluate and adapt on test data
# Each adapter can implement a different strategy (PTA, MultiProtoPTA, etc.).
# ─────────────────────────────────────────────────────────────────────────────


class BaseAdapter(ABC):
    """
    Abstract base class for all TTA (test-time adaptation) methods.

    This defines the interface that all test-time adaptation methods must follow.
    By inheriting from this class, new methods automatically integrate with runner.py
    without requiring any changes to the runner.

    To add a new method:
      1. Create models/<your_method>.py
      2. Subclass BaseAdapter and implement `run`.
      3. Pass --method <your_method> to runner.py.

    The runner will dynamically import models.<method>.build(cfg) to get
    an instance, then call adapter.run(...) for each dataset.

    Pattern: Strategy design pattern — each concrete adapter is a strategy
    for handling test-time adaptation, and runner selects which one to use.
    """

    def __init__(self, cfg: dict) -> None:
        """
        Initialize the adapter with a configuration dict.

        Args:
            cfg: dataset-specific config dict loaded from configs/<dataset>.yaml.
                 The adapter is re-instantiated (or cfg is updated) per dataset
                 so that per-dataset hyper-parameters are always fresh.

                 Typical keys: alpha, beta, T, K, tau_text, tau_proto, etc.
                 (specific keys depend on the adapter implementation).
        """
        # Store config as instance variable for later lookup in run()
        self.cfg = cfg

    @abstractmethod
    def run(
        self,
        loader: Any,
        clip_model: Any,
        clip_weights: Any,
        dataset_name: str,
    ) -> float:
        """
        Run one full evaluation pass over `loader`, adapting in real-time.

        This is the main entry point where the adapter processes test samples.
        Most implementations follow this pattern:
          1. For each test sample:
             a. Encode the image via CLIP → get image feature
             b. Compute initial CLIP logits (zero-shot)
             c. Update running prototype/model state (online adaptation)
             d. Compute final logits (possibly fused with updated model)
             e. Record accuracy

        Args:
            loader:       PyTorch DataLoader iterating over the test split.
                         Yields (image_batch, label_batch) tuples.
                         Note: batch_size=1 is hardcoded (TTA processes one at a time).

            clip_model:   Pre-loaded CLIP model (already on CUDA, eval mode).
                         Call clip_model.encode_image(images) to get features,
                         or use get_clip_logits() helper from utils.py.

            clip_weights: Pre-computed text-feature matrix of shape (D, C) on CUDA.
                         D = feature dimension (e.g. 512 for CLIP-ViT).
                         C = number of classes.
                         Transpose to get [C, D] for cosine similarity against images.

            dataset_name: Human-readable dataset name (e.g., "caltech101").
                         Used for logging and writing to outputs/result.txt.

        Returns:
            Top-1 accuracy (float, 0-100). Should also write results to
            outputs/result.txt as a side effect (see PTA for example).
        """
        raise NotImplementedError


def build(cfg: dict) -> "BaseAdapter":
    """
    Convenience factory — each models/<method>.py should override this at
    module level so the runner can call `models.<method>.build(cfg)`.

    Pattern: Factory pattern. The runner does NOT directly instantiate the
    adapter class; instead it calls the build() factory from the method module.
    This allows flexible initialization logic.

    Example in models/pta.py:
      def build(cfg: dict) -> PTAAdapter:
          return PTAAdapter(cfg)

    Usage in runner.py:
      import models
      adapter_module = __import__(f'models.{args.method}', fromlist=['build'])
      adapter = adapter_module.build(cfg)

    Raises:
        NotImplementedError: if called directly on this base class.
    """
    raise NotImplementedError
