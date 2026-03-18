"""
Official implementation of the pose-aware ViViT-based seizure classifier.

This package exposes:
- Data utilities for building clip tables and caching pose tubelets/tokens.
- Lightweight OpenPose helpers.
- The joint-attention classifier (frozen ViViT variant).
- The ViViT + LoRA end-to-end variant.
"""

from . import data, pose, models, metrics  # noqa: F401

