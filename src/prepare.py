"""Rebuild and verify every image array the GPU stage reads, without rerunning the slow mask stages.

Needs the extracted public datasets (scripts/download_data.sh) and the tracked tables in data/work/,
including the packed U2-Net masks (u2net_masks_packed.npy), which replace the ~1.5 h U2-Net CPU pass.

  python -m src.prepare prepare   rebuilds 224 frames, 4-class and 6-class variants and DTD textures,
                                  then checks MD5 checksums against data/work/array_checksums.json
  python -m src.prepare verify    checksum check of the rebuilt arrays only
  python -m src.prepare smoke     pre-fetches ImageNet weights, trains every architecture for one epoch on a
                                  small subset and checks every evaluation set (GPU recommended)
"""
from __future__ import annotations

import hashlib
import json
import sys

import numpy as np

from .common import WORK

CHECKED = [
    "trashnet_frames.npy",
    "realwaste_frames.npy",
    "dtd_train_256.npy",
    "variants/train_alpha.npy",
    "variants/train_ONLY_BG_T.npy",
    "variants/train_ONLY_BG_FIXED.npy",
    "variants/val_ONLY_BG_T.npy",
    "variants/test_ONLY_BG_FIXED.npy",
    "variants/test_base_MIXED_RAND_0.npy",
    "variants6/train_fg_alpha.npy",
    "variants6/train_ONLY_BG_T.npy",
    "variants6/val_ONLY_BG_T.npy",
    "variants6/test_base_ORIGINAL.npy",
    "variants6/test_base_MIXED_RAND_0.npy",
    "variants6/test_ONLY_BG_CORNERS.npy",
]
MASKS_U2 = WORK / "u2net_masks_packed.npy"
SUMS = WORK / "array_checksums.json"


def md5(path):
    return hashlib.md5(np.ascontiguousarray(np.load(path, mmap_mode="r")).tobytes()).hexdigest()


def prepare():
    from . import regimes, s01_inventory, s05_variants, s05_variants6

    if not (WORK / "trashnet_frames.npy").exists():
        _, f = s01_inventory.build_trashnet()
        np.save(WORK / "trashnet_frames.npy", f)
        _, f = s01_inventory.build_realwaste()
        np.save(WORK / "realwaste_frames.npy", f)
    if not (WORK / "variants" / "test_base_ORIGINAL.npy").exists():
        if not MASKS_U2.exists():
            sys.exit(f"{MASKS_U2} missing: restore it from the repository or run `python -m src.s04_masks u2net`.")
        m = np.unpackbits(np.load(MASKS_U2), axis=-1, count=224).astype(np.float16)
        np.save(WORK / "u2net_frame_saliency.npy", m)
        s05_variants.main()
    if not (WORK / "variants6" / "test_base_ORIGINAL.npy").exists():
        s05_variants6.main()
    regimes.dtd_train_textures()
    verify()


def verify():
    want = json.load(open(SUMS))
    bad = [f for f in CHECKED if not (WORK / f).exists() or md5(WORK / f) != want[f]]
    print("checksums:", "ALL MATCH" if not bad else f"MISMATCH in {bad}")
    if bad:
        sys.exit(1)


def smoke(n=64):
    import time

    import torch

    from .regimes import Compositor
    from .s08_cnn import ARCHS, DEVICE, build_model_and_opt, predict, train_model
    from .testsets import eval_sets, labels, probe_train_sets, tn_split

    print("device:", DEVICE, torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "")
    meta, xtr = tn_split("train")
    ytr = labels(meta)
    vmeta, xva = tn_split("val")
    yva = labels(vmeta)
    rng = np.random.default_rng(0)
    for regime in ("jitter", "paste6", "rand6", "shuffle6"):
        comp = Compositor(meta, xtr, regime)
        idx0 = int(np.flatnonzero(comp.el_idx >= 0)[0])
        assert comp(idx0, rng).shape == (224, 224, 3), regime

    # Pre-cache ImageNet weights & smoke-test all 4 architectures
    sub = meta.iloc[:n].reset_index(drop=True)
    comp = Compositor(sub, xtr[:n], "rand6")
    timings = {}
    for arch in ARCHS:
        t0 = time.time()
        build_model_and_opt(arch=arch, pretrained=True)
        model, _, _, _ = train_model(xtr[:n], ytr[:n], xva[:32], yva[:32], comp, seed=0, max_epochs=1, arch=arch, pretrained=True)
        p = predict(model, xva[:16])
        assert p.shape == (16, 6) and np.allclose(p.sum(1), 1.0, atol=1e-3), arch
        timings[arch] = time.time() - t0
        print(f"  arch {arch:12s} OK ({timings[arch]:.1f}s incl. weight check)")

    (pxtr, pytr), _ = probe_train_sets("corners")
    train_model(pxtr[:32], pytr[:32], pxtr[:32], pytr[:32], lambda k, r: pxtr[k], seed=0, max_epochs=1, arch="cnn")
    for name, x, _ in eval_sets(include_probe=True, include_v4=True, include_v6=True):
        assert x.shape[1:] == (224, 224, 3), name
    print("SMOKE TEST OK: all 4 architectures, 6-class variants and corner probe verified")


if __name__ == "__main__":
    {"prepare": prepare, "verify": verify, "smoke": smoke}[sys.argv[1]]()
