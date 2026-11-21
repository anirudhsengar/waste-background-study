"""Shared paths, constants and image helpers for every stage of the pipeline."""
from __future__ import annotations

import os
import random
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
EXT = DATA / "ext"
WORK = DATA / "work"          # derived artefacts (manifests, masks, variants, features)
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"
for _d in (WORK, RESULTS, FIGS):
    _d.mkdir(parents=True, exist_ok=True)

TRASHNET_DIR = EXT / "trashnet" / "dataset-resized"
REALWASTE_DIR = EXT / "realwaste" / "realwaste-main" / "RealWaste"
SEG_DIR = EXT / "seg" / "trashnet - Copy"
DTD_DIR = EXT / "dtd" / "dtd" / "images"
U2NET_ONNX = DATA / "raw" / "u2net.onnx"

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
# RealWaste folder -> shared class. Unlisted folders are excluded.
REALWASTE_MAP = {
    "Cardboard": "cardboard", "Glass": "glass", "Metal": "metal",
    "Paper": "paper", "Plastic": "plastic", "Miscellaneous Trash": "trash",
}

FRAME = 224
GREY = np.array([124, 116, 104], dtype=np.uint8)   # ImageNet mean in 0-255 (zero after normalisation)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
N_PAIRINGS = 5


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def read_rgb(path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"cannot read {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def crop_box(h: int, w: int) -> tuple[int, int, int, int]:
    """Centre square kept by the canonical transform, in original pixel coordinates (y0, y1, x0, x1)."""
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return y0, y0 + s, x0, x0 + s


def to_frame(arr: np.ndarray, interp=cv2.INTER_AREA) -> np.ndarray:
    """Canonical geometry: centre-crop the short side square, resize to 224 x 224.

    Identical to 'resize short side to 224 then centre-crop 224' up to rounding.
    Works for images (H, W, 3) and masks (H, W) alike.
    """
    h, w = arr.shape[:2]
    y0, y1, x0, x1 = crop_box(h, w)
    return cv2.resize(arr[y0:y1, x0:x1], (FRAME, FRAME), interpolation=interp)


def mask_to_frame(mask: np.ndarray) -> np.ndarray:
    """Binary mask (H, W) in original geometry -> binary uint8 mask in the 224 frame."""
    m = to_frame(mask.astype(np.float32), interp=cv2.INTER_AREA)
    return (m >= 0.5).astype(np.uint8)
