"""Shared training-step helpers for CLI training scripts."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .metrics import binary_metrics_for_assessment
from .models import compute_joint_padding_mask


def run_epoch_tokens_joint_classifier(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    *,
    train_mode: bool,
    zero_positional_emb: bool,
) -> dict:
    """One epoch over cached (tokens, pos, y) batches."""
    model.train() if train_mode else model.eval()
    all_logits = []
    all_y = []
    total_loss = 0.0
    n = 0

    for tokens, pos, y in loader:
        tokens = tokens.to(device, non_blocking=True)
        pos = pos.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        jmask = compute_joint_padding_mask(pos)
        if zero_positional_emb:
            pos = torch.zeros_like(pos)

        with torch.set_grad_enabled(train_mode):
            logits = model(tokens, pos, joint_padding_mask=jmask)
            loss = criterion(logits, y)
            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        total_loss += float(loss.item()) * y.numel()
        n += y.numel()
        all_logits.append(logits.detach().cpu())
        all_y.append(y.detach().cpu())

    logits_np = torch.cat(all_logits).numpy()
    y_np = torch.cat(all_y).numpy()
    mets = binary_metrics_for_assessment(logits_np, y_np, default_thr=0.5)
    mets["loss"] = total_loss / max(n, 1)
    return mets


def run_epoch_e2e_vivit_joint(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.amp.GradScaler,
    *,
    train_mode: bool,
    zero_positional_emb: bool,
) -> dict:
    """One epoch over cached (tubelets, pos, y) batches."""
    model.train() if train_mode else model.eval()
    all_logits = []
    all_y = []
    total_loss = 0.0
    n = 0
    use_cuda_amp = device.type == "cuda"

    for tubelets, pos, y in loader:
        tubelets = tubelets.to(device, non_blocking=True)
        pos = pos.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        jmask = compute_joint_padding_mask(pos)
        if zero_positional_emb:
            pos = torch.zeros_like(pos)

        with torch.set_grad_enabled(train_mode):
            with torch.amp.autocast("cuda", enabled=use_cuda_amp):
                logits = model(tubelets, pos, joint_padding_mask=jmask)
                loss = criterion(logits, y)
            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        total_loss += float(loss.item()) * y.numel()
        n += y.numel()
        all_logits.append(logits.detach().cpu())
        all_y.append(y.detach().cpu())

    logits_np = torch.cat(all_logits).numpy()
    y_np = torch.cat(all_y).numpy()
    mets = binary_metrics_for_assessment(logits_np, y_np, default_thr=0.5)
    mets["loss"] = total_loss / max(n, 1)
    return mets
