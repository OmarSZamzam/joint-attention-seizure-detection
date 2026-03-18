"""
Pose estimation and joint-tubelet utilities.

This module wraps the lightweight OpenPose-style network and provides:

- `ensure_lightweight_openpose_repo` and `load_pose_net` to obtain the pose model.
- `RandomClipSampler` to pick labeled clips and decode video frames.
- `run_pose_on_frames_rgb` to run pose on a sequence of frames.
- `build_joint_tubelets_from_poses` to convert keypoints into per-joint tubelets.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Video clip sampling / decoding
# ---------------------------------------------------------------------------


@dataclass
class RandomClipSampler:
    """
    Efficient sampler for (phase/state -> random row -> decode [start,end] -> downsample fps).

    Assumes `clips_df` has columns:
    - phase
    - video_path
    - clip_start_s
    - clip_end_s
    """

    clips_df: pd.DataFrame
    seed: int = 0

    def __post_init__(self):
        self.df = self.clips_df.reset_index(drop=True)
        self.rng = np.random.default_rng(self.seed)

        if "phase" not in self.df.columns:
            raise ValueError("clips_df must have a 'phase' column.")

        self.state_to_idx: Dict[str, np.ndarray] = {
            state: grp.index.to_numpy() for state, grp in self.df.groupby("phase", sort=False)
        }

        required = ["video_path", "clip_start_s", "clip_end_s"]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"clips_df missing required columns: {missing}")

    def sample_row(self, state: str) -> pd.Series:
        idxs = self.state_to_idx.get(state, None)
        if idxs is None or len(idxs) == 0:
            raise ValueError(f"No rows for state={state!r}. Available: {list(self.state_to_idx.keys())}")
        ridx = int(self.rng.choice(idxs))
        return self.df.iloc[ridx]

    @staticmethod
    def _open_video(video_path: str) -> cv2.VideoCapture:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        return cap

    @staticmethod
    def _get_fps(cap: cv2.VideoCapture, default_fps: float = 30.0) -> float:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0 or np.isnan(fps):
            return float(default_fps)
        return float(fps)

    @staticmethod
    def _seek_to_time(cap: cv2.VideoCapture, t_sec: float) -> None:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t_sec) * 1000.0)

    def decode_clip_downsample(
        self,
        row: pd.Series,
        out_fps: float = 6.0,
        convert_bgr_to_rgb: bool = True,
        max_frames: Optional[int] = None,
    ) -> np.ndarray:
        video_path = str(row["video_path"])
        start_s = float(row["clip_start_s"])
        end_s = float(row["clip_end_s"])

        cap = self._open_video(video_path)
        fps_in = self._get_fps(cap)

        skip = max(1, int(round(fps_in / float(out_fps))))
        self._seek_to_time(cap, start_s)

        frames: List[np.ndarray] = []
        kept = 0
        read_idx = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            t_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            if t_msec is None or np.isnan(t_msec):
                t_cur = start_s + (read_idx / fps_in)
            else:
                t_cur = float(t_msec) / 1000.0

            if t_cur >= end_s:
                break

            if (read_idx % skip) == 0:
                if convert_bgr_to_rgb:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                kept += 1
                if max_frames is not None and kept >= max_frames:
                    break

            read_idx += 1

        cap.release()

        if len(frames) == 0:
            raise RuntimeError(
                f"Decoded 0 frames for clip [{start_s:.3f}, {end_s:.3f}] from {video_path} "
                f"(fps_in={fps_in:.3f}, skip={skip})"
            )
        return np.stack(frames, axis=0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Lightweight OpenPose helpers
# ---------------------------------------------------------------------------


def ensure_lightweight_openpose_repo(
    repo_dir: str | Path,
    repo_url: str = "https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch.git",
) -> Path:
    """
    Clone the lightweight human pose estimation repository if missing and add to `sys.path`.
    """
    repo_dir = Path(repo_dir)
    if not repo_dir.exists():
        subprocess.check_call(["git", "clone", repo_url, str(repo_dir)])
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
    return repo_dir


def _strip_prefix(state_dict, prefix: str = "module."):
    return { (k[len(prefix):] if k.startswith(prefix) else k): v for k, v in state_dict.items() }


def load_pose_net(
    repo_dir: str | Path,
    checkpoint_path: str | Path,
    device: str = "cuda",
):
    """
    Load the lightweight OpenPose model and checkpoint.

    Returns `(net, info)` where:
    - `net` is a `PoseEstimationWithMobileNet` on the requested device.
    - `info` collects missing/unexpected keys from the checkpoint load.
    """
    ensure_lightweight_openpose_repo(repo_dir)

    import torch
    from models.with_mobilenet import PoseEstimationWithMobileNet  # type: ignore

    net = PoseEstimationWithMobileNet()

    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        elif "model" in ckpt:
            sd = ckpt["model"]
        elif "net" in ckpt:
            sd = ckpt["net"]
        elif all(hasattr(v, "shape") for v in ckpt.values()):
            sd = ckpt
        else:
            raise ValueError(f"Unrecognized checkpoint dict keys: {list(ckpt.keys())[:30]}")
    else:
        sd = ckpt

    sd = _strip_prefix(sd, "module.")
    missing, unexpected = net.load_state_dict(sd, strict=False)

    net = net.to(device).eval()
    return net, {"missing": missing, "unexpected": unexpected}


# ---------------------------------------------------------------------------
# Pose inference on frames
# ---------------------------------------------------------------------------


def _pose_score(p) -> float:
    return float(getattr(p, "confidence", getattr(p, "score", 0.0)))


def pick_main_pose(poses):
    if not poses:
        return None
    return max(poses, key=_pose_score)


def infer_poses_bgr(
    img_bgr: np.ndarray,
    net,
    device: str,
    input_height: int = 256,
    stride: int = 8,
    upsample_ratio: int = 4,
    pad_value=(0, 0, 0),
):
    """
    Run lightweight OpenPose inference on a single BGR image.
    Returns a list of `Pose` objects.
    """
    import torch
    from modules.keypoints import extract_keypoints, group_keypoints  # type: ignore
    from modules.pose import Pose  # type: ignore
    from val import normalize, pad_width  # type: ignore

    orig_h = img_bgr.shape[0]
    scale = input_height / orig_h
    resized = cv2.resize(img_bgr, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    normalized = normalize(resized, img_mean=(128, 128, 128), img_scale=1 / 256)

    min_dims = [input_height, max(resized.shape[1], input_height)]
    padded, pad = pad_width(normalized, stride, pad_value, min_dims=min_dims)

    tensor_img = torch.from_numpy(padded).permute(2, 0, 1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        stages_output = net(tensor_img)

    heatmaps = stages_output[-2].squeeze(0).permute(1, 2, 0).cpu().numpy()
    pafs = stages_output[-1].squeeze(0).permute(1, 2, 0).cpu().numpy()

    heatmaps = cv2.resize(heatmaps, None, fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)
    pafs = cv2.resize(pafs, None, fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    total_keypoints_num = 0
    all_keypoints_by_type = []
    for kpt_idx in range(Pose.num_kpts):
        total_keypoints_num += extract_keypoints(
            heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num
        )

    pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)

    # heatmap coords -> original pixel coords
    all_keypoints[:, 0] = (all_keypoints[:, 0] * stride / upsample_ratio - pad[1]) / scale
    all_keypoints[:, 1] = (all_keypoints[:, 1] * stride / upsample_ratio - pad[0]) / scale

    poses = []
    for n in range(len(pose_entries)):
        if len(pose_entries[n]) == 0:
            continue
        pose_keypoints = np.full((Pose.num_kpts, 2), -1, dtype=np.int32)
        for kpt_id in range(Pose.num_kpts):
            kpt_ind = int(pose_entries[n][kpt_id])
            if kpt_ind != -1:
                x, y = all_keypoints[kpt_ind, 0], all_keypoints[kpt_ind, 1]
                pose_keypoints[kpt_id] = (int(round(x)), int(round(y)))
        poses.append(Pose(pose_keypoints, pose_entries[n][-1]))

    return poses


def run_pose_on_frames_rgb(
    frames_rgb: np.ndarray,
    net,
    device: str,
    input_height: int = 256,
    keep: str = "main",
    do_tracking: bool = True,
):
    """
    Run pose on a sequence of RGB frames:

    - `frames_rgb`: (N, H, W, 3) uint8
    - `keep`: "main" (default) to keep the highest-confidence pose per frame.
    """
    from modules.pose import track_poses  # type: ignore

    prev_poses = []
    outputs = []

    for i in range(len(frames_rgb)):
        frame_bgr = cv2.cvtColor(frames_rgb[i], cv2.COLOR_RGB2BGR)
        poses = infer_poses_bgr(frame_bgr, net, device, input_height=input_height)

        if do_tracking:
            track_poses(prev_poses, poses, smooth=True)
            prev_poses = poses

        if keep == "main":
            outputs.append(pick_main_pose(poses))
        elif keep == "all":
            outputs.append(poses)
        else:
            raise ValueError("keep must be 'main' or 'all'")

    return outputs


# ---------------------------------------------------------------------------
# Tubelet construction
# ---------------------------------------------------------------------------

BODY_PLUS_NOSE_JOINTS = list(range(0, 14))  # keep nose + body; drop eyes/ears


def _crop_square_patch_rgb(
    frame_rgb: np.ndarray,
    cx: int,
    cy: int,
    P: int,
    pad_mode: str = "constant",
    pad_value: int = 0,
) -> np.ndarray:
    """
    Crop a P×P patch centered at (cx, cy). Pads if the patch goes out of bounds.
    Returns uint8 RGB patch of shape (P, P, 3).
    """
    H, W, _ = frame_rgb.shape
    half = P // 2

    x0 = cx - half
    y0 = cy - half
    x1 = x0 + P
    y1 = y0 + P

    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x1 - W)
    pad_bottom = max(0, y1 - H)

    if pad_left or pad_top or pad_right or pad_bottom:
        if pad_mode == "constant":
            padded = cv2.copyMakeBorder(
                frame_rgb,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                borderType=cv2.BORDER_CONSTANT,
                value=(pad_value, pad_value, pad_value),
            )
        elif pad_mode == "reflect":
            padded = cv2.copyMakeBorder(
                frame_rgb,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                borderType=cv2.BORDER_REFLECT_101,
            )
        else:
            raise ValueError("pad_mode must be 'constant' or 'reflect'.")

        # shift coords into padded image
        x0 += pad_left
        x1 += pad_left
        y0 += pad_top
        y1 += pad_top

        patch = padded[y0:y1, x0:x1]
    else:
        patch = frame_rgb[y0:y1, x0:x1]

    if patch.shape[0] != P or patch.shape[1] != P:
        patch = cv2.resize(patch, (P, P), interpolation=cv2.INTER_LINEAR)
    return patch.astype(np.uint8)


def build_joint_tubelets_from_poses(
    frames_rgb: np.ndarray,  # (T,H,W,3) uint8
    poses_out,  # list length T; each element is Pose or None
    P: int = 120,
    joint_indices: Optional[List[int]] = None,
    pad_mode: str = "constant",
    pad_value: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct per-joint tubelets given pose keypoints and frames.

    Returns:
      - tubelets: (J, T, 3, P, P) uint8
      - pos:      (J, T, 3) float32, last dim = (x, y, joint_id), x=y=-1 if missing.
    """
    assert frames_rgb.ndim == 4 and frames_rgb.shape[-1] == 3, "frames_rgb must be (T,H,W,3)"
    T = frames_rgb.shape[0]

    if joint_indices is None:
        joint_indices = BODY_PLUS_NOSE_JOINTS

    J = len(joint_indices)

    tubelets = np.zeros((J, T, 3, P, P), dtype=np.uint8)
    pos = np.zeros((J, T, 3), dtype=np.float32)

    # Fill joint IDs
    for j, jid in enumerate(joint_indices):
        pos[j, :, 2] = float(jid)

    for t in range(T):
        frame = frames_rgb[t]
        pose = poses_out[t] if poses_out is not None else None

        if pose is None:
            for j in range(J):
                pos[j, t, 0] = -1.0
                pos[j, t, 1] = -1.0
            continue

        kpts = pose.keypoints

        for j, jid in enumerate(joint_indices):
            x, y = kpts[jid]
            if x < 0 or y < 0:
                pos[j, t, 0] = -1.0
                pos[j, t, 1] = -1.0
                continue

            cx, cy = int(x), int(y)
            pos[j, t, 0] = float(cx)
            pos[j, t, 1] = float(cy)

            patch = _crop_square_patch_rgb(
                frame, cx=cx, cy=cy, P=P, pad_mode=pad_mode, pad_value=pad_value
            )  # (P,P,3) RGB

            # store as (3,P,P)
            tubelets[j, t] = patch.transpose(2, 0, 1)

    return tubelets, pos

