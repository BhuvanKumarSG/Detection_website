"""inspect_ckpt.py
Quick inspection tool for PyTorch / Lightning checkpoints.
Usage:
  python inspect_ckpt.py --ckpt PATH_TO_CHECKPOINT

It prints top-level keys, whether it contains 'state_dict', some example state_dict keys
and tensor shapes to help determine the model architecture and name prefixes.
"""
import argparse
import torch
from pathlib import Path

def inspect_ckpt(path):
    print(f"Loading checkpoint: {path}")
    try:
        # Attempt a safe weights-only load (PyTorch 2.6+ uses weights_only=True by default).
        ckpt = torch.load(path, map_location='cpu')
    except Exception as e:
        # Fallback to a full load when weights-only fails. This may execute arbitrary code
        # from the checkpoint, so only do this for trusted files.
        print("Initial torch.load failed:", e)
        print("Retrying torch.load with weights_only=False (only for trusted checkpoints)...")
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
    print("Top-level type:", type(ckpt))
    if isinstance(ckpt, dict):
        keys = list(ckpt.keys())
        print("Top-level keys:")
        for k in keys:
            print("  ", k)
    else:
        print("Checkpoint is not a dict; it's a raw state_dict or tensor.")

    # Determine where state dict is
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state = ckpt['state_dict']
        print("Found 'state_dict' with", len(state), "keys")
    elif isinstance(ckpt, dict) and 'model' in ckpt and isinstance(ckpt['model'], dict):
        state = ckpt['model']
        print("Found 'model' entry with", len(state), "keys")
    elif isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        state = ckpt
        print("Top-level dict appears to be a state_dict with", len(state), "keys")
    else:
        print("Could not locate a state_dict automatically. Showing repr of object.")
        print(repr(ckpt)[:1000])
        return

    # Show some state_dict key examples and shapes
    sample_keys = list(state.keys())[:30]
    print('\nSample state_dict keys and shapes (up to 30):')
    for k in sample_keys:
        v = state[k]
        if isinstance(v, torch.Tensor):
            print(f"  {k} -> {tuple(v.shape)}")
        else:
            print(f"  {k} -> {type(v)}")

    # Heuristic: common prefixes
    prefixes = {}
    for k in state.keys():
        prefix = k.split('.')[0]
        prefixes[prefix] = prefixes.get(prefix, 0) + 1
    print('\nTop prefixes (most common):')
    top_prefixes = sorted(prefixes.items(), key=lambda x: -x[1])[:10]
    for p, c in top_prefixes:
        print(f"  {p}: {c}")

    # If there is training metadata
    if isinstance(ckpt, dict):
        meta_keys = [k for k in ckpt.keys() if k not in ('state_dict','model')]
        if meta_keys:
            print('\nOther metadata keys:')
            for k in meta_keys:
                print('  ', k)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True, help='Path to checkpoint file (.ckpt, .pt, .pth)')
    args = p.parse_args()
    path = Path(args.ckpt)
    if not path.exists():
        print(f'File not found: {path}')
        raise SystemExit(2)
    inspect_ckpt(str(path))
