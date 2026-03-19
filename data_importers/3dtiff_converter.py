#!/usr/bin/env python3
"""Split 3D TIFF stacks into per-slice directories for Noise2Inverse360."""
import tifffile
import numpy as np
from pathlib import Path

base = Path("/data2/vnikitin/alcf/brain/Y350a_dist1234/denoising/03132026/obj_complex")

for fname, outdir in [
    ("delta_even.tiff", "delta_even"),
    ("delta_odd.tiff",  "delta_odd"),
    ("delta_all.tiff",  "delta_all"),
]:
    vol = tifffile.imread(base / fname)  # [Z, H, W]
    out = base / outdir
    out.mkdir(exist_ok=True)
    print(f"{fname}: {vol.shape} -> {out}")
    for i, sl in enumerate(vol):
        tifffile.imwrite(out / f"{i:05d}.tiff", sl)
