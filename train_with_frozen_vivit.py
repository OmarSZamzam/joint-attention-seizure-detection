"""
Train the proposed joint-attention seizure classifier with a **frozen** ViViT encoder.

This script trains only the joint-attention transformer on top of per-joint ViViT tokens.
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from seizure_classifier.data import (
    RandomClipSampler,
    cache_joint_tokens_to_dir,
    CachedJointTokenDataset,
    collate_cached_joint_tokens,
)
from seizure_classifier.models import JointTransformerClassifier
from seizure_classifier.pose import ensure_lightweight_openpose_repo, load_pose_net
from seizure_classifier.training_loops import run_epoch_tokens_joint_classifier


def parse_args():
    p = argparse.ArgumentParser(description="Train with frozen ViViT + joint attention")
    p.add_argument(
        "--clips_csv",
        type=str,
        required=True,
        help=(
            "CSV with precomputed fixed-length clips (e.g. 5s segments). "
            "Must include at least: patient_id, phase, video_path, clip_start_s, clip_end_s."
        ),
    )
    p.add_argument(
        "--cache_dir",
        type=str,
        required=True,
        help="Directory to store/load cached joint tokens (.npz files and manifests).",
    )
    p.add_argument("--pose_repo", type=str, default="lightweight-human-pose-estimation.pytorch")
    p.add_argument(
        "--pose_ckpt",
        type=str,
        default="model_weights/pose.pth",
        help="Path to lightweight OpenPose checkpoint (default: model_weights/pose.pth).",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--val_frac", type=float, default=0.4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument(
        "--vivit_model",
        type=str,
        default="google/vivit-b-16x2-kinetics400",
        help="HuggingFace model id for the frozen ViViT backbone.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--skip_cache",
        action="store_true",
        help=(
            "If set, assume tokens are already cached under --cache_dir and do not "
            "re-run pose+ViViT. This is purely an efficiency toggle; the method "
            "itself always uses a frozen ViViT."
        ),
    )
    p.add_argument(
        "--disable_positional_emb",
        action="store_true",
        help=(
            "If set, zero out all joint positional embeddings before feeding them to the model "
            "(pos = torch.zeros_like(pos))."
        ),
    )
    return p.parse_args()


def train():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    clips_df = pd.read_csv(args.clips_csv)
    phase_to_label = {"interictal": 0, "ictal": 1}
    clips_df = clips_df[clips_df["phase"].isin(phase_to_label.keys())].reset_index(drop=True)

    patients = clips_df["patient_id"].unique().tolist()
    rng = np.random.default_rng(args.seed)
    rng.shuffle(patients)

    n_val = max(1, int(len(patients) * args.val_frac))
    val_patients = set(patients[:n_val])

    is_val = clips_df["patient_id"].isin(val_patients).to_numpy()
    train_idx = np.where(~is_val)[0]
    val_idx = np.where(is_val)[0]

    print("patients:", len(patients), "train_patients:", len(patients) - n_val, "val_patients:", n_val)
    print("train clips:", len(train_idx), "val clips:", len(val_idx))
    print("train phase counts:\n", clips_df.iloc[train_idx]["phase"].value_counts())
    print("val phase counts:\n", clips_df.iloc[val_idx]["phase"].value_counts())

    sampler = RandomClipSampler(clips_df, seed=args.seed)

    if not args.skip_cache:
        ensure_lightweight_openpose_repo(args.pose_repo)
        pose_net, _ = load_pose_net(args.pose_repo, args.pose_ckpt, device=str(device))
        pose_net.eval()

        _ = cache_joint_tokens_to_dir(
            clips_df=clips_df,
            cache_dir=args.cache_dir,
            sampler=sampler,
            pose_net=pose_net,
            pose_device=str(device),
            vivit_model_name=args.vivit_model,
            vivit_device=str(device),
            phase_to_label=phase_to_label,
            split="train",
            indices=train_idx,
        )

        _ = cache_joint_tokens_to_dir(
            clips_df=clips_df,
            cache_dir=args.cache_dir,
            sampler=sampler,
            pose_net=pose_net,
            pose_device=str(device),
            vivit_model_name=args.vivit_model,
            vivit_device=str(device),
            phase_to_label=phase_to_label,
            split="val",
            indices=val_idx,
        )
    else:
        print("Skipping cache construction; expecting precomputed manifests in --cache_dir.")

    train_manifest_path = os.path.join(args.cache_dir, "manifest_train.csv")
    val_manifest_path = os.path.join(args.cache_dir, "manifest_val.csv")

    train_ds = CachedJointTokenDataset(train_manifest_path)
    val_ds = CachedJointTokenDataset(val_manifest_path)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_cached_joint_tokens,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_cached_joint_tokens,
    )

    example_tokens, _, _ = next(iter(train_loader))
    token_dim = example_tokens.shape[-1]

    model = JointTransformerClassifier(
        token_dim=token_dim,
        d_model=256,
        nhead=8,
        num_layers=4,
        dropout=0.5,
        use_cls_token=True,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit = nn.BCEWithLogitsLoss()

    zero_pos = args.disable_positional_emb
    best_auprc = -1.0
    for epoch in range(args.epochs):
        train_m = run_epoch_tokens_joint_classifier(
            model, train_loader, device, opt, crit, train_mode=True, zero_positional_emb=zero_pos
        )
        val_m = run_epoch_tokens_joint_classifier(
            model, val_loader, device, opt, crit, train_mode=False, zero_positional_emb=zero_pos
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train loss {train_m['loss']:.4f} auroc {train_m['auroc']:.3f} auprc {train_m['auprc']:.3f} | "
            f"val loss {val_m['loss']:.4f} auroc {val_m['auroc']:.3f} auprc {val_m['auprc']:.3f} | "
            f"Acc@0.5 {val_m['acc']:.3f} F1@0.5 {val_m['f1']:.3f}"
        )

        if val_m["auprc"] > best_auprc:
            best_auprc = val_m["auprc"]
            ckpt_path = os.path.join(args.cache_dir, "frozen_vivit_joint_clf_best.pt")
            torch.save({"epoch": epoch, "model": model.state_dict(), "metrics": val_m}, ckpt_path)
            print("Saved best checkpoint to", ckpt_path)


if __name__ == "__main__":
    train()
