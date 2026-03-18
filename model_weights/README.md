# Pretrained weights

Place released checkpoints here so paths stay predictable:

| File | Role |
|------|------|
| `pose.pth` | Lightweight OpenPose (human-pose) weights for tubelets |
| Classifier `.pt` | **Frozen ViViT + joint attention only**. Two model weights: trained **with** positional embeddings, and trained **without** (same as `--disable_positional_emb`). |

Training defaults assume **`model_weights/pose.pth`** when you run scripts from the repository root. Override with `--pose_ckpt` / your own paths if needed.

## Classifier checkpoint format

Checkpoints are saved as:

```python
torch.save(
    {
        "best_thr": float(val_m["best_thr"]),  # threshold for binary decision
        "model": model.state_dict(),
    },
    path,
)
```

### Loading for inference

```python
import torch

ckpt = torch.load("model_weights/your_classifier.pt", map_location="cpu")
state_dict = ckpt["model"]
best_thr = float(ckpt.get("best_thr", 0.5))

# Build the same architecture as training, then:
model.load_state_dict(state_dict, strict=True)
model.eval()
# probs = torch.sigmoid(logits); pred = (probs >= best_thr).long()
```

The architecture must match **`train_with_frozen_vivit.py`**: `JointTransformerClassifier` on top of frozen ViViT tokens. At inference, use positional inputs consistently with how that checkpoint was trained (with real joint positions vs. zeros if it was trained with `--disable_positional_emb`).

