"""Training regimes (`Compositor`): how each training image is produced per epoch.

  orig      T-orig     original image
  jitter    T-jitter   colour jitter (brightness, contrast, saturation 0.4, hue 0.1) with probability 0.5
  paste6    T-paste    object (polygon mask) pasted onto the background-only version of a same-class training
                       image from another duplicate group, probability 0.5
  shuffle6  T-shuffle  as paste6, but the donor class is drawn uniformly, independent of the object's class
  rand6     T-rand     object pasted onto a random 224 x 224 crop of a DTD texture, probability 0.5
  paste     T-paste-U2Net (suppl.), rand  T-rand-U2Net (suppl.): the same with U2-Net masks, four classes

`one_copy_each(seed)` gives one augmented copy of every eligible image (used for the SVMs).
"""
from __future__ import annotations

import json

import cv2
import numpy as np
import pandas as pd

from .common import DTD_DIR, FRAME, WORK
from .s05_variants import VDIR, composite
from .s05_variants6 import V6DIR

DTD_SIZE = 256


def dtd_split():
    cats = sorted(p.name for p in DTD_DIR.iterdir() if p.is_dir())
    perm = np.random.default_rng(0).permutation(len(cats))
    split = {"train": sorted(cats[i] for i in perm[:37]), "heldout": sorted(cats[i] for i in perm[37:])}
    json.dump(split, open(WORK / "dtd_category_split.json", "w"), indent=1)
    return split


def dtd_train_textures():
    path = WORK / "dtd_train_256.npy"
    if not path.exists():
        out = []
        for c in dtd_split()["train"]:
            for f in sorted((DTD_DIR / c).glob("*.jpg")):
                img = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
                h, w = img.shape[:2]
                s = min(h, w)
                img = img[(h - s) // 2 : (h - s) // 2 + s, (w - s) // 2 : (w - s) // 2 + s]
                out.append(cv2.resize(img, (DTD_SIZE, DTD_SIZE), interpolation=cv2.INTER_AREA))
        np.save(path, np.stack(out))
    return np.load(path, mmap_mode="r")


def jitter_rgb(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Photometric jitter matching torchvision ColorJitter(0.4, 0.4, 0.4, 0.1) in uint8 RGB."""
    x = img.astype(np.float32)
    # Brightness
    x = x * rng.uniform(0.6, 1.4)
    # Contrast around per-channel mean
    mean = x.mean(axis=(0, 1), keepdims=True)
    x = (x - mean) * rng.uniform(0.6, 1.4) + mean
    x = np.clip(x, 0, 255).astype(np.uint8)
    # Saturation & Hue in OpenCV HSV (H in [0, 180))
    hsv = cv2.cvtColor(x, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-18.0, 18.0)) % 180.0
    hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.6, 1.4), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


class Compositor:
    """Builds the augmented training image for row k of the training split."""

    def __init__(self, train_meta: pd.DataFrame, train_frames: np.ndarray, regime: str, p: float = 0.5):
        self.regime, self.p, self.frames = regime, p, train_frames
        if regime == "orig":
            return
        if regime == "jitter":
            self.el_idx = np.arange(len(train_meta))
            return

        if regime in ("paste6", "rand6", "shuffle6"):
            fg = pd.read_csv(V6DIR / "train_fg_meta.csv")
            pos = {i: k for k, i in enumerate(fg.id)}
            self.el_idx = np.array([pos.get(i, -1) for i in train_meta.id])
            self.alpha = np.load(V6DIR / "train_fg_alpha.npy").astype(np.float32)
            if regime == "shuffle6":
                cf = pd.read_csv(V6DIR / "train_cf_meta.csv")
                self.bgs = np.load(V6DIR / "train_ONLY_BG_T.npy", mmap_mode="r")
                classes = sorted(cf.cls.unique())
                self.pools = []  # per foreground image: one donor pool per class, own duplicate group excluded
                for g in fg.group:
                    self.pools.append([np.flatnonzero((cf.cls.values == c) & (cf.group.values != g)) for c in classes])
            elif regime == "paste6":
                cf = pd.read_csv(V6DIR / "train_cf_meta.csv")
                self.bgs = np.load(V6DIR / "train_ONLY_BG_T.npy", mmap_mode="r")
                self.pools = []
                for c, g in zip(fg.cls, fg.group):
                    p_idx = np.flatnonzero((cf.cls.values == c) & (cf.group.values != g))
                    if len(p_idx) == 0:
                        p_idx = np.flatnonzero(cf.cls.values == c)
                    self.pools.append(p_idx)
            else:
                self.tex = dtd_train_textures()
            return

        # Original 4-class U2-Net regimes ("paste", "rand")
        el = pd.read_csv(VDIR / "train_eligible_meta.csv")
        pos = {i: k for k, i in enumerate(el.id)}
        self.el_idx = np.array([pos.get(i, -1) for i in train_meta.id])
        self.alpha = np.load(VDIR / "train_alpha.npy").astype(np.float32)
        if regime == "paste":
            self.bgs = np.load(VDIR / "train_ONLY_BG_T.npy", mmap_mode="r")
            self.pools = [np.flatnonzero((el.cls.values == c) & (el.group.values != g)) for c, g in zip(el.cls, el.group)]
        elif regime == "rand":
            self.tex = dtd_train_textures()
        else:
            raise ValueError(f"Unknown regime: {regime}")

    def __call__(self, k: int, rng: np.random.Generator) -> np.ndarray:
        img = self.frames[k]
        if self.regime == "orig":
            return np.asarray(img)
        if self.regime == "jitter":
            if rng.random() >= self.p:
                return np.asarray(img)
            return jitter_rgb(np.asarray(img), rng)
        e = self.el_idx[k]
        if e < 0 or rng.random() >= self.p:
            return np.asarray(img)
        if self.regime in ("paste", "paste6"):
            bg = self.bgs[rng.choice(self.pools[e])].astype(np.float32)
        elif self.regime == "shuffle6":
            pools = self.pools[e]
            bg = self.bgs[rng.choice(pools[rng.integers(len(pools))])].astype(np.float32)
        else:
            t = self.tex[rng.integers(len(self.tex))]
            y, x = rng.integers(0, DTD_SIZE - FRAME + 1, 2)
            bg = np.asarray(t[y : y + FRAME, x : x + FRAME], dtype=np.float32)
            if rng.random() < 0.5:
                bg = bg[:, ::-1]
        return composite(np.asarray(img, np.float32), self.alpha[e], bg)

    def one_copy_each(self, seed: int) -> tuple[np.ndarray, np.ndarray]:
        """For the SVMs: one augmented copy of every eligible image (always augmented), seed-fixed."""
        rng = np.random.default_rng(seed)
        keep_p, self.p = self.p, 1.0
        idx = np.flatnonzero(self.el_idx >= 0)
        out = np.stack([self(int(k), rng) for k in idx])
        self.p = keep_p
        return idx, out
