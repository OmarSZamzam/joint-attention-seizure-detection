"""
Data utilities for the seizure video dataset.

This module assumes you already have a clip-level dataframe for fixed-length
video segments (e.g. 5 s clips). It does **not** contain any logic for constructing those segments.

Given such a dataframe, it provides:
- Caching utilities for:
  - Joint tubelets (for end-to-end ViViT + LoRA training).
  - Precomputed ViViT joint tokens (for frozen-encoder training).
- PyTorch datasets and collate functions for cached samples.
"""

from __future__ import annotations

import os
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .pose import (
    RandomClipSampler,
    run_pose_on_frames_rgb,
    build_joint_tubelets_from_poses,
)


# ---------------------------------------------------------------------------
# Padding utilities
# ---------------------------------------------------------------------------

def pad_or_trim_frames(frames_rgb: np.ndarray, k_target: int) -> np.ndarray:
    """
    Ensure a fixed temporal length `k_target` by trimming or repeating the last frame.
    """
    T = frames_rgb.shape[0]
    if T == k_target:
        return frames_rgb
    if T > k_target:
        return frames_rgb[:k_target]
    pad_n = k_target - T
    last = frames_rgb[-1:]
    pad = np.repeat(last, pad_n, axis=0)
    return np.concatenate([frames_rgb, pad], axis=0)


def pad_or_trim_list(xs: List, k_target: int) -> List:
    """
    List version of pad_or_trim_frames.
    """
    T = len(xs)
    if T == k_target:
        return xs
    if T > k_target:
        return xs[:k_target]
    last = xs[-1] if T > 0 else None
    return xs + [last] * (k_target - T)


# ---------------------------------------------------------------------------
# Joint tubelet caching (for end-to-end ViViT + LoRA training)
# ---------------------------------------------------------------------------

def cache_joint_tubelets_to_dir(
    clips_df: pd.DataFrame,
    cache_dir: str,
    sampler: RandomClipSampler,
    pose_net,
    pose_device: str,
    phase_to_label: Dict[str, int],
    split: str,
    indices: np.ndarray,
    out_fps: float = 6.0,
    clip_len_s: float = 5.0,
    P: int = 120,
    input_height: int = 256,
    do_tracking: bool = True,
    overwrite: bool = False,
    log_every: int = 20,
) -> pd.DataFrame:
    """
    Precompute tubelets for each clip and write one `.npz` per sample:

    - tubelets: (J, T, 3, P, P) uint8
    - pos:      (J, T, 3) float32
    - y:        scalar float32
    """
    os.makedirs(cache_dir, exist_ok=True)
    split_dir = os.path.join(cache_dir, split)
    os.makedirs(split_dir, exist_ok=True)

    rows: List[Dict] = []
    t0 = time.time()
    n_ok = 0
    n_fail = 0

    for n, idx in enumerate(tqdm(indices.tolist(), desc=f"cache_tubelets_{split}")):
        r = clips_df.iloc[idx]
        phase = str(r["phase"])
        if phase not in phase_to_label:
            continue

        y = float(phase_to_label[phase])

        patient = str(r.get("patient_id", "NA"))
        seizure = str(r.get("seizure_id", "NA"))
        start_s = float(r.get("clip_start_s", -1))
        fname = f"{idx:08d}_{patient}_{seizure}_{phase}_{start_s:.3f}.npz".replace(":", "-")
        out_path = os.path.join(split_dir, fname)

        if (not overwrite) and os.path.exists(out_path):
            rows.append(
                {
                    "idx": idx,
                    "split": split,
                    "path": out_path,
                    "y": y,
                    "phase": phase,
                    "patient_id": patient,
                    "seizure_id": seizure,
                    "clip_start_s": start_s,
                }
            )
            n_ok += 1
            continue

        try:
            k_target = int(round(clip_len_s * out_fps))

            frames = sampler.decode_clip_downsample(
                r, out_fps=out_fps, convert_bgr_to_rgb=True
            )
            frames = pad_or_trim_frames(frames, k_target)

            poses_out = run_pose_on_frames_rgb(
                frames_rgb=frames,
                net=pose_net,
                device=pose_device,
                input_height=input_height,
                keep="main",
                do_tracking=do_tracking,
            )
            poses_out = pad_or_trim_list(poses_out, k_target)

            tubelets, pos = build_joint_tubelets_from_poses(
                frames_rgb=frames,
                poses_out=poses_out,
                P=P,
            )

            np.savez_compressed(
                out_path,
                tubelets=tubelets.astype(np.uint8),
                pos=pos.astype(np.float32),
                y=np.float32(y),
            )

            rows.append(
                {
                    "idx": idx,
                    "split": split,
                    "path": out_path,
                    "y": y,
                    "phase": phase,
                    "patient_id": patient,
                    "seizure_id": seizure,
                    "clip_start_s": start_s,
                }
            )
            n_ok += 1

        except Exception as e:  # pragma: no cover - logging path
            n_fail += 1
            print(e)
            rows.append(
                {
                    "idx": idx,
                    "split": split,
                    "path": "",
                    "y": y,
                    "phase": phase,
                    "patient_id": patient,
                    "seizure_id": seizure,
                    "clip_start_s": start_s,
                    "error": repr(e),
                }
            )

        if (n + 1) % log_every == 0:
            dt = time.time() - t0
            print(
                f"[cache_tubelets:{split}] {n+1}/{len(indices)} "
                f"ok={n_ok} fail={n_fail} elapsed={dt/60:.1f}min"
            )

    manifest = pd.DataFrame(rows)
    manifest_path = os.path.join(cache_dir, f"manifest_{split}.csv")
    manifest.to_csv(manifest_path, index=False)
    print(f"[cache_tubelets:{split}] wrote manifest: {manifest_path}")
    print(f"[cache_tubelets:{split}] ok={n_ok} fail={n_fail}")
    return manifest


class CachedTubeletDataset(Dataset):
    """
    Dataset over cached tubelet `.npz` files produced by `cache_joint_tubelets_to_dir`.

    Each sample returns:
    - tubelets: (J, T, 3, P, P) uint8
    - pos:      (J, T, 3) float32
    - y:        scalar float32
    """

    def __init__(self, manifest_csv: str):
        self.manifest = pd.read_csv(manifest_csv)
        self.manifest = (
            self.manifest.dropna(subset=["path"])
            .loc[self.manifest["path"].astype(str).str.len() > 0]
            .reset_index(drop=True)
        )

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, i: int):
        path = self.manifest.loc[i, "path"]
        data = np.load(str(path))
        return (
            data["tubelets"].astype(np.uint8),
            data["pos"].astype(np.float32),
            np.float32(data["y"]),
        )


def collate_cached_tubelets(batch):
    tubelets = torch.from_numpy(np.stack([b[0] for b in batch], axis=0))  # (B,J,T,3,P,P)
    pos = torch.from_numpy(np.stack([b[1] for b in batch], axis=0))  # (B,J,T,3)
    y = torch.from_numpy(np.stack([b[2] for b in batch], axis=0))  # (B,)
    return tubelets, pos, y


# ---------------------------------------------------------------------------
# Joint-token caching (for frozen ViViT encoder training)
# ---------------------------------------------------------------------------

from .models import extract_joint_tokens_vivit  # noqa: E402


def cache_joint_tokens_to_dir(
    clips_df: pd.DataFrame,
    cache_dir: str,
    sampler: RandomClipSampler,
    pose_net,
    pose_device: str,
    vivit_model_name: str,
    vivit_device: str,
    phase_to_label: Dict[str, int],
    split: str,
    indices: np.ndarray,
    out_fps: float = 6.0,
    clip_len_s: float = 5.0,
    P: int = 120,
    vivit_num_frames: int = 32,
    vivit_out_size: int = 224,
    input_height: int = 256,
    do_tracking: bool = True,
    overwrite: bool = False,
    log_every: int = 20,
) -> pd.DataFrame:
    """
    Precompute ViViT joint tokens with a **frozen** encoder and write one `.npz` per sample:

    - tokens: (J, Dtok) float32
    - pos:    (J, T, 3) float32
    - y:      scalar float32
    """
    os.makedirs(cache_dir, exist_ok=True)
    split_dir = os.path.join(cache_dir, split)
    os.makedirs(split_dir, exist_ok=True)

    rows: List[Dict] = []
    t0 = time.time()
    n_ok = 0
    n_fail = 0

    for n, idx in enumerate(tqdm(indices.tolist(), desc=f"cache_tokens_{split}")):
        r = clips_df.iloc[idx]
        phase = str(r["phase"])
        if phase not in phase_to_label:
            continue

        y = float(phase_to_label[phase])

        patient = str(r.get("patient_id", "NA"))
        seizure = str(r.get("seizure_id", "NA"))
        start_s = float(r.get("clip_start_s", -1))
        fname = f"{idx:08d}_{patient}_{seizure}_{phase}_{start_s:.3f}.npz".replace(":", "-")
        out_path = os.path.join(split_dir, fname)

        if (not overwrite) and os.path.exists(out_path):
            rows.append(
                {
                    "idx": idx,
                    "split": split,
                    "path": out_path,
                    "y": y,
                    "phase": phase,
                    "patient_id": patient,
                    "seizure_id": seizure,
                    "clip_start_s": start_s,
                }
            )
            n_ok += 1
            continue

        try:
            k_target = int(round(clip_len_s * out_fps))

            frames = sampler.decode_clip_downsample(
                r, out_fps=out_fps, convert_bgr_to_rgb=True
            )
            frames = pad_or_trim_frames(frames, k_target)

            poses_out = run_pose_on_frames_rgb(
                frames_rgb=frames,
                net=pose_net,
                device=pose_device,
                input_height=input_height,
                keep="main",
                do_tracking=do_tracking,
            )
            poses_out = pad_or_trim_list(poses_out, k_target)

            tubelets, pos = build_joint_tubelets_from_poses(
                frames_rgb=frames,
                poses_out=poses_out,
                P=P,
            )

            tokens = extract_joint_tokens_vivit(
                tubelets=tubelets,
                model_name=vivit_model_name,
                device=vivit_device,
                num_frames=vivit_num_frames,
                out_size=vivit_out_size,
                pool="cls",
                batch_size=2,
            )

            np.savez_compressed(
                out_path,
                tokens=tokens.astype(np.float32),
                pos=pos.astype(np.float32),
                y=np.float32(y),
            )

            rows.append(
                {
                    "idx": idx,
                    "split": split,
                    "path": out_path,
                    "y": y,
                    "phase": phase,
                    "patient_id": patient,
                    "seizure_id": seizure,
                    "clip_start_s": start_s,
                }
            )
            n_ok += 1

        except Exception as e:  # pragma: no cover - logging path
            n_fail += 1
            print(e)
            rows.append(
                {
                    "idx": idx,
                    "split": split,
                    "path": "",
                    "y": y,
                    "phase": phase,
                    "patient_id": patient,
                    "seizure_id": seizure,
                    "clip_start_s": start_s,
                    "error": repr(e),
                }
            )

        if (n + 1) % log_every == 0:
            dt = time.time() - t0
            print(
                f"[cache_tokens:{split}] {n+1}/{len(indices)} "
                f"ok={n_ok} fail={n_fail} elapsed={dt/60:.1f}min"
            )

    manifest = pd.DataFrame(rows)
    manifest_path = os.path.join(cache_dir, f"manifest_{split}.csv")
    manifest.to_csv(manifest_path, index=False)
    print(f"[cache_tokens:{split}] wrote manifest: {manifest_path}")
    print(f"[cache_tokens:{split}] ok={n_ok} fail={n_fail}")
    return manifest


class CachedJointTokenDataset(Dataset):
    """
    Dataset over cached joint tokens `.npz` produced by `cache_joint_tokens_to_dir`.

    Each sample returns:
    - tokens: (J, Dtok) float32
    - pos:    (J, T, 3) float32
    - y:      scalar float32
    """

    def __init__(self, manifest_csv: str):
        self.manifest = pd.read_csv(manifest_csv)
        self.manifest = (
            self.manifest.dropna(subset=["path"])
            .loc[self.manifest["path"].astype(str).str.len() > 0]
            .reset_index(drop=True)
        )

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, i: int):
        path = self.manifest.loc[i, "path"]
        data = np.load(str(path))
        return (
            data["tokens"].astype(np.float32),
            data["pos"].astype(np.float32),
            np.float32(data["y"]),
        )


def collate_cached_joint_tokens(batch):
    tokens = torch.from_numpy(np.stack([b[0] for b in batch], axis=0))  # (B,J,Dtok)
    pos = torch.from_numpy(np.stack([b[1] for b in batch], axis=0))  # (B,J,T,3)
    y = torch.from_numpy(np.stack([b[2] for b in batch], axis=0))  # (B,)
    return tokens, pos, y

