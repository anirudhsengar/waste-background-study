"""Every evaluation set, by name, with its labels (index into CLASSES)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import CLS2IDX, N_PAIRINGS, WORK
from .s05_variants import VDIR
from .s05_variants6 import V6DIR

VARIANTS = ["ORIGINAL", "ONLY_FG", "ONLY_BG_T"] + [f"MIXED_{k}_{p}" for k in ("SAME", "RAND") for p in range(N_PAIRINGS)]
MASK_KINDS = ["base", "erode3", "dilate3"]


def split_meta():
    return pd.read_csv(WORK / "split.csv")


def labels(meta):
    return meta.cls.map(CLS2IDX).values


def tn_split(split, col="split"):
    meta = split_meta()
    meta = meta[meta[col] == split].reset_index(drop=True)
    frames = np.load(WORK / "trashnet_frames.npy", mmap_mode="r")
    return meta, np.asarray(frames[meta.row.values])


def eval_sets(include_probe=False, include_v4=True, include_v6=True):
    """Yield (name, images, labels). Images are uint8 (N, 224, 224, 3)."""
    for split in ("val", "test"):
        meta, x = tn_split(split)
        yield f"tn_{split}", x, labels(meta)
    rw = pd.read_csv(WORK / "realwaste_manifest.csv")
    yield "rw", np.load(WORK / "realwaste_frames.npy", mmap_mode="r"), labels(rw)

    if include_v4 and (VDIR / "test_eligible_meta.csv").exists():
        tm = pd.read_csv(VDIR / "test_eligible_meta.csv")
        y = labels(tm)
        for kind in MASK_KINDS:
            for v in VARIANTS:
                if kind != "base" and v == "ORIGINAL":
                    continue
                yield f"var_{kind}_{v}", np.load(VDIR / f"test_{kind}_{v}.npy", mmap_mode="r"), y
        if include_probe:
            keep = np.load(VDIR / "test_fixedbox_keep.npy")
            yield "probe_fixed", np.load(VDIR / "test_ONLY_BG_FIXED.npy", mmap_mode="r"), y[keep]

    if include_v6 and (V6DIR / "test_cf_meta.csv").exists():
        tm6 = pd.read_csv(V6DIR / "test_cf_meta.csv")
        y6 = labels(tm6)
        for kind in MASK_KINDS:
            for v in VARIANTS:
                if kind != "base" and v == "ORIGINAL":
                    continue
                yield f"var6_{kind}_{v}", np.load(V6DIR / f"test_{kind}_{v}.npy", mmap_mode="r"), y6
        if include_probe:
            cm6 = pd.read_csv(V6DIR / "test_corner_meta.csv")
            yield "probe_corners", np.load(V6DIR / "test_ONLY_BG_CORNERS.npy", mmap_mode="r"), labels(cm6)


def probe_train_sets(kind):
    """(train, val) images and labels for Probe A:
    - 'bg' / 'fixed': 4-class U2-Net ONLY-BG-T and fixed-box control
    - 'bg6': 6-class human-polygon ONLY-BG-T (1,201 train / 255 val)
    - 'corners': 6-class untouched 16x16 corner patches (810 train / 182 val)
    """
    out = []
    if kind in ("bg", "fixed"):
        for split in ("train", "val"):
            m = pd.read_csv(VDIR / f"{split}_eligible_meta.csv")
            y = labels(m)
            if kind == "bg":
                out.append((np.load(VDIR / f"{split}_ONLY_BG_T.npy"), y))
            else:
                keep = np.load(VDIR / f"{split}_fixedbox_keep.npy")
                out.append((np.load(VDIR / f"{split}_ONLY_BG_FIXED.npy"), y[keep]))
    elif kind == "bg6":
        for split in ("train", "val"):
            m = pd.read_csv(V6DIR / f"{split}_cf_meta.csv")
            out.append((np.load(V6DIR / f"{split}_ONLY_BG_T.npy"), labels(m)))
    elif kind == "corners":
        for split in ("train", "val"):
            m = pd.read_csv(V6DIR / f"{split}_corner_meta.csv")
            out.append((np.load(V6DIR / f"{split}_ONLY_BG_CORNERS.npy"), labels(m)))
    else:
        raise ValueError(f"Unknown probe kind: {kind}")
    return out
