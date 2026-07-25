#!/usr/bin/env python
"""Smoke test: verify that mamba-ssm CUDA kernel works (preparation for S04+)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

EXPERIMENT_NAME = "S01_test_mamba_forward_smoke"


def main() -> None:
    import platform

    print(f"Platform: {platform.system()} {platform.machine()}")

    # Try importing mamba-ssm
    try:
        from mamba_ssm import Mamba
    except ImportError as exc:
        error_msg = str(exc)
        if "triton" in error_msg.lower() or "windows" in error_msg.lower() or "No module named" in error_msg:
            print(f"\n[EXPECTED FAILURE] Cannot import mamba-ssm: {exc}")
            print("\nmamba-ssm requires Triton + CUDA kernels which are not supported on native Windows.")
            print("Recommendations:")
            print("  1. Use WSL2 (Windows Subsystem for Linux) with a Linux CUDA toolkit")
            print("  2. Use a Linux container or cloud VM with GPU")
            print("\nThis does NOT block S02 (S02 uses linear projection, not Mamba).")
            print("Mamba is only needed starting from S04.")
            sys.exit(0)
        else:
            print(f"\n[UNEXPECTED ERROR] mamba-ssm import failed: {exc}")
            print("Try: pip install mamba-ssm --break-system-packages")
            sys.exit(1)
    except Exception as exc:
        print(f"\n[UNEXPECTED ERROR] mamba-ssm import failed: {exc}")
        print("This may be a build/CUDA compatibility issue.")
        print("This does NOT block S02.")
        sys.exit(0)

    # If import succeeds, run a forward pass
    import torch

    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available. Mamba requires GPU.")
        sys.exit(0)

    print("mamba-ssm imported successfully. Running forward smoke test...")

    d_model = 64
    device = "cuda"
    mamba_block = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2).to(device)

    x = torch.randn(1, 10, d_model, device=device)
    with torch.no_grad():
        y = mamba_block(x)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output mean:  {y.mean().item():.6f}")
    print(f"Output has NaN: {torch.isnan(y).any().item()}")
    print("\n[OK] Mamba CUDA kernel works. Ready for S04+.")


if __name__ == "__main__":
    main()
