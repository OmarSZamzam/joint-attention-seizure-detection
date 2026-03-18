"""
Model components for the pose-aware ViViT seizure classifier.

This module groups:
- Joint position embedding and transformer classifier.
- Frozen ViViT feature extractor for joint tubelets.
- ViViT + LoRA builders and end-to-end joint models.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import VivitModel


# ---------------------------------------------------------------------------
# Joint-attention classifier over ViViT tokens
# ---------------------------------------------------------------------------


def compute_joint_padding_mask(pos_t: torch.Tensor) -> torch.Tensor:
    """
    Missing joints use x = y = -1 in ``pos_t``.

    Args:
        pos_t: (B, J, T, 3).

    Returns:
        (B, J) bool; True means ignore that joint in attention.
    """
    xy = pos_t[..., :2]
    valid_any = (xy[..., 0] >= 0).any(dim=2)
    return ~valid_any


class JointPositionEmbedder(nn.Module):
    """
    Learnable network: (B, J, T, 3) -> (B, J, D).

    We treat the per-joint position sequence length `T` as time and embed it
    with an MLP per time step, then pool across time (mean pooling).
    """

    def __init__(self, d_model: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.frame_mlp = nn.Sequential(
            nn.Linear(3, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pos: (B, J, T, 3) with (x, y, id) per frame.
        Returns:
            (B, J, D) joint embeddings.
        """
        B, J, T, C = pos.shape
        assert C == 3

        e = self.frame_mlp(pos)  # (B, J, T, D)
        e = e.mean(dim=2)  # (B, J, D) over time
        e = self.out_norm(e)
        return e


class JointTransformerClassifier(nn.Module):
    """
    Transformer over joints:

      tokens: (B, J, Dtok) from ViViT
      pos:    (B, J, T, 3) raw joint positions
      fused:  tok_proj(tokens) + pos_embed(pos)

    A transformer encoder over sequence length J (+ optional CLS) produces
    a single binary logit per clip.
    """

    def __init__(
        self,
        token_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
        pos_hidden: int = 128,
        use_cls_token: bool = True,
    ):
        super().__init__()
        self.use_cls_token = use_cls_token

        self.tok_proj = nn.Linear(token_dim, d_model)
        self.pos_embedder = JointPositionEmbedder(d_model=d_model, hidden=pos_hidden, dropout=dropout)

        if use_cls_token:
            self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls, std=0.02)

        if dim_feedforward is None:
            dim_feedforward = 4 * d_model

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)  # binary logit

    def forward(
        self,
        tokens: torch.Tensor,  # (B, J, Dtok)
        pos: torch.Tensor,  # (B, J, T, 3)
        joint_padding_mask: Optional[torch.Tensor] = None,  # (B, J), True=ignore joint
    ) -> torch.Tensor:
        x = self.tok_proj(tokens)  # (B, J, D)
        e = self.pos_embedder(pos)  # (B, J, D)
        x = x + e

        if self.use_cls_token:
            B = x.size(0)
            cls = self.cls.expand(B, 1, -1)  # (B, 1, D)
            x = torch.cat([cls, x], dim=1)  # (B, 1+J, D)

            if joint_padding_mask is not None:
                cls_mask = torch.zeros((B, 1), dtype=torch.bool, device=x.device)
                key_padding_mask = torch.cat([cls_mask, joint_padding_mask], dim=1)
            else:
                key_padding_mask = None
        else:
            key_padding_mask = joint_padding_mask

        x = self.encoder(x, src_key_padding_mask=key_padding_mask)

        if self.use_cls_token:
            pooled = x[:, 0, :]
        else:
            pooled = x.mean(dim=1)

        pooled = self.norm(pooled)
        logit = self.head(pooled).squeeze(-1)
        return logit


# ---------------------------------------------------------------------------
# Frozen ViViT joint token extractor (non-differentiable)
# ---------------------------------------------------------------------------


def _uniform_frame_indices(T: int, num_frames: int) -> np.ndarray:
    if T <= 0:
        raise ValueError("T must be > 0")
    if num_frames <= 0:
        raise ValueError("num_frames must be > 0")
    if T == 1:
        return np.zeros((num_frames,), dtype=np.int64)
    idx = np.linspace(0, T - 1, num=num_frames, dtype=np.float32)
    return np.round(idx).astype(np.int64)


def extract_joint_tokens_vivit(
    tubelets: np.ndarray,  # (J, T, 3, P, P) uint8 recommended
    model_name: str = "google/vivit-b-16x2-kinetics400",
    device: str = "cuda",
    num_frames: Optional[int] = None,
    out_size: int = 224,
    pool: str = "cls",
    batch_size: int = 2,
    enforce_model_num_frames: bool = True,
) -> np.ndarray:
    """
    Pretrained ViViT token extractor that does NOT rely on AutoImageProcessor.

    Returns:
        tokens: (J, D) float32 numpy array.
    """
    assert pool in ("cls", "mean")
    assert tubelets.ndim == 5 and tubelets.shape[2] == 3, "tubelets must be (J,T,3,P,P)"

    model = VivitModel.from_pretrained(model_name).to(device).eval()

    expected = getattr(model.config, "num_frames", None)
    if num_frames is None:
        num_frames = int(expected) if expected is not None else 32

    if enforce_model_num_frames and expected is not None and int(num_frames) != int(expected):
        num_frames = int(expected)

    J, T, C, P, _ = tubelets.shape

    idx = _uniform_frame_indices(T, num_frames)
    tube_sel = tubelets[:, idx]  # (J, T, 3, P, P)

    x = torch.from_numpy(tube_sel)
    if x.dtype != torch.float32:
        x = x.float()
    if x.max() > 1.5:
        x = x / 255.0

    x2 = x.reshape(J * num_frames, 3, P, P)
    x2 = F.interpolate(x2, size=(out_size, out_size), mode="bilinear", align_corners=False)
    x = x2.reshape(J, num_frames, 3, out_size, out_size)

    mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype).view(1, 1, 3, 1, 1)
    x = (x - mean) / std

    tokens_all: List[torch.Tensor] = []

    with torch.no_grad():
        for s in range(0, J, batch_size):
            xb = x[s : s + batch_size].to(device)  # (B, T, C, H, W)
            outputs = model(pixel_values=xb)
            hs = outputs.last_hidden_state  # (B, N, D)

            if pool == "cls":
                tok = hs[:, 0, :]
            else:
                tok = hs.mean(dim=1)

            tokens_all.append(tok.detach().cpu())

    tokens = torch.cat(tokens_all, dim=0)  # (J, D)
    return tokens.float().numpy()


# ---------------------------------------------------------------------------
# ViViT + LoRA and end-to-end joint models
# ---------------------------------------------------------------------------


def vivit_joint_tokens_forward(
    tubelets: torch.Tensor,  # (B,J,T,3,P,P)
    vivit_model: Any,
    num_frames: Optional[int] = None,
    out_size: int = 224,
    pool: str = "cls",
    enforce_model_num_frames: bool = True,
) -> torch.Tensor:
    """
    Differentiable forward:
      tubelets (B,J,T,3,P,P) -> tokens (B,J,D).
    """
    assert pool in ("cls", "mean")
    assert tubelets.ndim == 6 and tubelets.shape[3] == 3, "tubelets must be (B,J,T,3,P,P)"

    B, J, T, C, P, _ = tubelets.shape
    device = tubelets.device
    dtype = torch.float32

    expected = getattr(vivit_model.config, "num_frames", None)
    if num_frames is None:
        num_frames = int(expected) if expected is not None else 32

    if enforce_model_num_frames and expected is not None and int(num_frames) != int(expected):
        num_frames = int(expected)

    x = tubelets
    if x.dtype != torch.float32:
        x = x.float()
    if x.max() > 1.5:
        x = x / 255.0

    if T == 1:
        idx = torch.zeros((num_frames,), device=device, dtype=torch.long)
    else:
        idx_f = torch.linspace(0, T - 1, steps=num_frames, device=device, dtype=torch.float32)
        idx = torch.round(idx_f).to(torch.long).clamp(0, T - 1)

    x = x.index_select(dim=2, index=idx)  # (B,J,T,3,P,P)

    T = num_frames
    x = x.reshape(B * J * T, 3, P, P).to(dtype=dtype)
    x = F.interpolate(x, size=(out_size, out_size), mode="bilinear", align_corners=False)
    x = x.reshape(B * J, T, 3, out_size, out_size)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=x.dtype).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=x.dtype).view(1, 1, 3, 1, 1)
    x = (x - mean) / std

    out = vivit_model(pixel_values=x)
    hs = out.last_hidden_state

    if pool == "cls":
        tok = hs[:, 0, :]
    else:
        tok = hs.mean(dim=1)

    tok = tok.reshape(B, J, -1)
    return tok


def vivit_joint_tokens_forward_chunked(
    tubelets: torch.Tensor,  # (B,J,T,3,P,P)
    vivit_model: Any,
    num_frames: Optional[int] = None,
    out_size: int = 224,
    pool: str = "cls",
    enforce_model_num_frames: bool = True,
    joint_chunk: int = 1,
    move_chunk_to_device: bool = True,
) -> torch.Tensor:
    """
    Memory-efficient variant of `vivit_joint_tokens_forward` that processes
    joints in micro-batches to reduce peak GPU memory, while preserving gradients.
    """
    assert pool in ("cls", "mean")
    assert tubelets.ndim == 6 and tubelets.shape[3] == 3, "tubelets must be (B,J,T,3,P,P)"
    B, J, T, C, P, _ = tubelets.shape

    try:
        model_device = next(vivit_model.parameters()).device
    except StopIteration:
        model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokens_chunks = []
    for j0 in range(0, J, joint_chunk):
        j1 = min(J, j0 + joint_chunk)
        tube_chunk = tubelets[:, j0:j1]  # (B, jc, T, 3, P, P)

        if move_chunk_to_device and tube_chunk.device != model_device:
            tube_chunk = tube_chunk.to(model_device, non_blocking=True)

        tok_chunk = vivit_joint_tokens_forward(
            tubelets=tube_chunk,
            vivit_model=vivit_model,
            num_frames=num_frames,
            out_size=out_size,
            pool=pool,
            enforce_model_num_frames=enforce_model_num_frames,
        )  # (B, jc, D)

        tokens_chunks.append(tok_chunk)

    return torch.cat(tokens_chunks, dim=1)  # (B,J,D)


def build_vivit_with_lora_last_n_layers(
    model_name: str = "google/vivit-b-16x2-kinetics400",
    last_n: int = 2,
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    device: str = "cuda",
):
    """
    Build a ViViT backbone with LoRA adapters applied only to the last `last_n`
    encoder layers, suitable for lightweight end-to-end finetuning.
    """
    try:
        from peft import LoraConfig, get_peft_model
    except Exception as e:  # pragma: no cover - import-time guard
        raise ImportError("peft is required for LoRA. Install with: pip install peft") from e

    model = VivitModel.from_pretrained(model_name)

    layer_ids = list(range(model.config.num_hidden_layers - last_n, model.config.num_hidden_layers))
    layer_prefixes = [f"encoder.layer.{i}." for i in layer_ids]

    candidate_suffixes = (
        ".attention.attention.query",
        ".attention.attention.key",
        ".attention.attention.value",
        ".attention.output.dense",
        ".intermediate.dense",
        ".output.dense",
    )

    existing = {name for name, _ in model.named_modules()}
    target_modules: List[str] = []
    for lp in layer_prefixes:
        for suf in candidate_suffixes:
            full = lp + suf.lstrip(".")
            if full in existing:
                target_modules.append(full)

    if len(target_modules) == 0:
        target_modules = ["query", "key", "value", "dense"]

    from peft import LoraConfig, get_peft_model  # type: ignore

    cfg = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=target_modules,
        task_type="FEATURE_EXTRACTION",
    )

    model = get_peft_model(model, cfg).to(device).train()
    return model


def enable_vivit_memory_savers(vivit_model: Any) -> Any:
    """
    Turn on common HF memory savers:
      - gradient checkpointing
      - disable cache
    Safe to call on PEFT-wrapped models as well.
    """
    m = vivit_model
    base = getattr(vivit_model, "base_model", None)

    for cand in [m, base]:
        if cand is None:
            continue
        if hasattr(cand, "gradient_checkpointing_enable"):
            try:
                cand.gradient_checkpointing_enable()
            except Exception:
                pass
        if hasattr(cand, "config") and hasattr(cand.config, "use_cache"):
            cand.config.use_cache = False

    return vivit_model


class End2EndVivitJointModelChunked(nn.Module):
    """
    End-to-end model:
      tubelets -> ViViT(+LoRA) tokens (chunked) -> JointTransformerClassifier.
    """

    def __init__(
        self,
        vivit_model,
        joint_classifier: nn.Module,
        vivit_num_frames: Optional[int] = None,
        vivit_out_size: int = 224,
        vivit_pool: str = "cls",
        enforce_model_num_frames: bool = True,
        joint_chunk: int = 1,
        move_chunk_to_device: bool = True,
    ):
        super().__init__()
        self.vivit = vivit_model
        self.clf = joint_classifier
        self.vivit_num_frames = vivit_num_frames
        self.vivit_out_size = vivit_out_size
        self.vivit_pool = vivit_pool
        self.enforce_model_num_frames = enforce_model_num_frames
        self.joint_chunk = joint_chunk
        self.move_chunk_to_device = move_chunk_to_device

    def forward(
        self,
        tubelets: torch.Tensor,  # (B,J,T,3,P,P)
        pos: torch.Tensor,  # (B,J,T,3)
        joint_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        tokens = vivit_joint_tokens_forward_chunked(
            tubelets=tubelets,
            vivit_model=self.vivit,
            num_frames=self.vivit_num_frames,
            out_size=self.vivit_out_size,
            pool=self.vivit_pool,
            enforce_model_num_frames=self.enforce_model_num_frames,
            joint_chunk=self.joint_chunk,
            move_chunk_to_device=self.move_chunk_to_device,
        )
        return self.clf(tokens, pos, joint_padding_mask=joint_padding_mask)

