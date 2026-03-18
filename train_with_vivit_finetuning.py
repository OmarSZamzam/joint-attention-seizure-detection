"""
Train the proposed joint-attention seizure classifier while finetuning ViViT with LoRA.
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
    cache_joint_tubelets_to_dir,
    CachedTubeletDataset,
    collate_cached_tubelets,
)
from seizure_classifier.models import (
    JointTransformerClassifier,
    End2EndVivitJointModelChunked,
    build_vivit_with_lora_last_n_layers,
    enable_vivit_memory_savers,
)
from seizure_classifier.pose import ensure_lightweight_openpose_repo, load_pose_net
from seizure_classifier.training_loops import run_epoch_e2e_vivit_joint


def parse_args():
    p = argparse.ArgumentParser(description="Train with ViViT finetuning + joint attention (LoRA)")
    p.add_argument(
        "--clips_csv",
        type=str,
        required=True,
        help="CSV with precomputed fixed-length clips (5s segments), "
        "including columns: patient_id, phase, video_path, clip_start_s, clip_end_s.",
    )
    p.add_argument("--cache_dir", type=str, required=True, help="Directory to store tubelet cache")
    p.add_argument("--pose_repo", type=str, default="lightweight-human-pose-estimation.pytorch")
    p.add_argument(
        "--pose_ckpt",
        type=str,
        default="model_weights/pose.pth",
        help="Path to lightweight OpenPose checkpoint (default: model_weights/pose.pth).",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--val_frac", type=float, default=0.4)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--vivit_model", type=str, default="google/vivit-b-16x2-kinetics400")
    p.add_argument("--lora_last_n", type=int, default=8)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
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

    phase_to_label = {"interictal": 0, "ictal": 1}

    clips_df = pd.read_csv(args.clips_csv)
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

    ensure_lightweight_openpose_repo(args.pose_repo)
    pose_net, _ = load_pose_net(args.pose_repo, args.pose_ckpt, device=str(device))
    pose_net.eval()

    _ = cache_joint_tubelets_to_dir(
        clips_df=clips_df,
        cache_dir=args.cache_dir,
        sampler=sampler,
        pose_net=pose_net,
        pose_device=str(device),
        phase_to_label=phase_to_label,
        split="train",
        indices=train_idx,
    )

    _ = cache_joint_tubelets_to_dir(
        clips_df=clips_df,
        cache_dir=args.cache_dir,
        sampler=sampler,
        pose_net=pose_net,
        pose_device=str(device),
        phase_to_label=phase_to_label,
        split="val",
        indices=val_idx,
    )

    train_ds = CachedTubeletDataset(os.path.join(args.cache_dir, "manifest_train.csv"))
    val_ds = CachedTubeletDataset(os.path.join(args.cache_dir, "manifest_val.csv"))

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_cached_tubelets,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_cached_tubelets,
    )

    vivit = build_vivit_with_lora_last_n_layers(
        model_name=args.vivit_model,
        last_n=args.lora_last_n,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        device=str(device),
    )
    enable_vivit_memory_savers(vivit)

    token_dim = int(vivit.config.hidden_size)
    joint_clf = JointTransformerClassifier(
        token_dim=token_dim,
        d_model=256,
        nhead=8,
        num_layers=vivit.config.num_hidden_layers,
        dropout=0.5,
        use_cls_token=False,
    ).to(device)

    model = End2EndVivitJointModelChunked(
        vivit_model=vivit,
        joint_classifier=joint_clf,
        vivit_num_frames=32,
        vivit_out_size=224,
        vivit_pool="cls",
        enforce_model_num_frames=True,
        joint_chunk=1,
        move_chunk_to_device=True,
    ).to(device)

    base = getattr(vivit, "base_model", vivit)
    base = getattr(base, "model", base)
    try:
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        base.config.use_cache = False
    except Exception:
        pass

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    crit = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    zero_pos = args.disable_positional_emb
    best_auprc = -1.0
    for epoch in range(args.epochs):
        train_m = run_epoch_e2e_vivit_joint(
            model,
            train_loader,
            device,
            opt,
            crit,
            scaler,
            train_mode=True,
            zero_positional_emb=zero_pos,
        )
        val_m = run_epoch_e2e_vivit_joint(
            model,
            val_loader,
            device,
            opt,
            crit,
            scaler,
            train_mode=False,
            zero_positional_emb=zero_pos,
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train loss {train_m['loss']:.4f} auroc {train_m['auroc']:.3f} auprc {train_m['auprc']:.3f} | "
            f"val loss {val_m['loss']:.4f} auroc {val_m['auroc']:.3f} auprc {val_m['auprc']:.3f} | "
            f"Acc@0.5 {val_m['acc']:.3f} F1@0.5 {val_m['f1']:.3f} "
            f"(best_thr {val_m['best_thr']:.2f}, F1@best {val_m['best_f1']:.3f})"
        )

        if val_m["auprc"] > best_auprc:
            best_auprc = val_m["auprc"]
            ckpt_path = os.path.join(args.cache_dir, "vivit_lora_joint_clf_best.pt")
            torch.save({"epoch": epoch, "model": model.state_dict(), "metrics": val_m}, ckpt_path)
            print("Saved best checkpoint to", ckpt_path)


if __name__ == "__main__":
    train()
