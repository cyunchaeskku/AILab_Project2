"""Average the EMA generator weights across several checkpoints.

Why this exists:
    Late in a stabilized training run, consecutive checkpoints sit in the same
    loss basin but at slightly different points. Averaging their weights
    (a cheap "weight-space ensemble", like SWA) often nudges FID down a little
    for free -- no extra training, no extra params at inference.

Caveat:
    Only average NEARBY checkpoints from the SAME stage (e.g. the last few of a
    stabilize phase). Averaging far-apart or different-resolution checkpoints
    mixes different solutions and usually hurts.

The merged checkpoint reuses the latest input checkpoint's metadata, so it loads
exactly like any training checkpoint:
    load_generator(MERGED_PATH)   # from verify_truncation

Run from the repo root:
    python average_ckpts.py
"""
from __future__ import annotations

from pathlib import Path

import torch


# ----------------------------- configuration ----------------------------- #
# Nearby late checkpoints from the SAME stabilize stage, oldest -> newest.
CKPT_PATHS = [
    Path("/content/drive/MyDrive/OpenAILab/project2/pggan_1024_v2/ckpt_000425176.pt"),
    Path("/content/drive/MyDrive/OpenAILab/project2/pggan_1024_v2/ckpt_000450176.pt"),
    Path("/content/drive/MyDrive/OpenAILab/project2/pggan_1024_v2/ckpt_000475176.pt"),
]
MERGED_PATH = Path(
    "/content/drive/MyDrive/OpenAILab/project2/pggan_1024_v2/ckpt_avg_512.pt"
)
EMA_KEY = "G_ema_state"
# ------------------------------------------------------------------------- #


def average_state_dicts(state_dicts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Mean of matching tensors across checkpoints.

    Float tensors are averaged; non-float tensors (e.g. int buffers) are taken
    from the last checkpoint unchanged, since averaging them is meaningless.
    """
    reference = state_dicts[-1]
    averaged: dict[str, torch.Tensor] = {}
    for key, ref_tensor in reference.items():
        if not torch.is_floating_point(ref_tensor):
            averaged[key] = ref_tensor.clone()
            continue
        stacked = torch.stack(
            [sd[key].float() for sd in state_dicts], dim=0
        )                                                   # (N, *param_shape)
        averaged[key] = stacked.mean(dim=0).to(ref_tensor.dtype)  # (*param_shape)
    return averaged


def main() -> None:
    if len(CKPT_PATHS) < 2:
        raise ValueError("Need at least 2 checkpoints to average")

    checkpoints = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in CKPT_PATHS
    ]
    for path, ckpt in zip(CKPT_PATHS, checkpoints):
        if EMA_KEY not in ckpt:
            raise KeyError(f"{path.name} has no '{EMA_KEY}'")

    ema_states = [ckpt[EMA_KEY] for ckpt in checkpoints]
    keys = set(ema_states[0].keys())
    for path, state in zip(CKPT_PATHS, ema_states):
        if set(state.keys()) != keys:
            raise ValueError(f"{path.name} EMA keys differ -- not the same architecture")

    averaged_ema = average_state_dicts(ema_states)

    # Reuse the latest checkpoint as the template, swap in the averaged EMA.
    merged = checkpoints[-1]
    merged[EMA_KEY] = averaged_ema

    MERGED_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, MERGED_PATH)
    print(f"Averaged {len(CKPT_PATHS)} checkpoints -> {MERGED_PATH}")
    print("Inputs:")
    for path in CKPT_PATHS:
        print(f"  {path.name}")
    print("Measure it with verify_fid.py (point CKPT_PATH at the merged file).")


if __name__ == "__main__":
    main()
