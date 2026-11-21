"""Stage 5: four-class edited test images in the 224 frame (U2-Net masks; supplementary suite).

Test set (eligible test images), for mask versions base / erode3 / dilate3:
  ORIGINAL, ONLY_FG, ONLY_BG_T, MIXED_SAME[p], MIXED_RAND[p] for pairings p = 0..4.
Background-only probe sets (train / val / test eligible): ONLY_BG_T, and the fixed-box control.
Also the per-image paste alpha and ONLY_BG_T of training images, used by the four-class paste and rand regimes.
"""
import json

import cv2
import numpy as np
import pandas as pd

from .common import CLASSES, FRAME, GREY, N_PAIRINGS, RESULTS, WORK

ERODE_PASTE, FEATHER_SIGMA, DILATE_REMOVE = 2, 1.5, 8
VDIR = WORK / "variants"
VDIR.mkdir(exist_ok=True)


def _disk(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def perturb(mask, kind):
    if kind == "erode3":
        return cv2.erode(mask, _disk(3))
    if kind == "dilate3":
        return cv2.dilate(mask, _disk(3))
    return mask


def paste_alpha(mask):
    """Erode 2 px to strip background fringe, feather with a Gaussian (sigma 1.5)."""
    a = cv2.erode(mask, _disk(ERODE_PASTE)).astype(np.float32)
    return cv2.GaussianBlur(a, (0, 0), FEATHER_SIGMA)


def removal_box(mask):
    d = cv2.dilate(mask, _disk(DILATE_REMOVE))
    ys, xs = np.nonzero(d)
    return ys.min(), ys.max() + 1, xs.min(), xs.max() + 1


def _mirror_tile(strip, n, axis):
    """Tile `strip` along `axis` to length n, alternating flipped/unflipped copies so edges match."""
    parts, total, flip = [], 0, True
    while total < n:
        parts.append(np.flip(strip, axis) if flip else strip)
        total += strip.shape[axis]; flip = not flip
    return np.concatenate(parts, axis)[:n] if axis == 0 else np.concatenate(parts, axis)[:, :n]


def fill_box_tiled(img, box):
    """Replace `box` with mirror-tiled copies of the largest background strip outside it
    (adapted from Xiao et al.'s ONLY-BG-T)."""
    y0, y1, x0, x1 = box
    out = img.copy()
    cands = {"top": y0 * FRAME, "bottom": (FRAME - y1) * FRAME, "left": x0 * FRAME, "right": (FRAME - x1) * FRAME}
    side = max(cands, key=cands.get)
    h, w = y1 - y0, x1 - x0
    if side == "top":       # rows above the box, read downwards from the box edge
        out[y0:y1, x0:x1] = _mirror_tile(img[:y0, x0:x1], h, 0)
    elif side == "bottom":
        out[y0:y1, x0:x1] = _mirror_tile(img[y1:, x0:x1][::-1], h, 0)[::-1]
    elif side == "left":
        out[y0:y1, x0:x1] = _mirror_tile(img[y0:y1, :x0], w, 1)
    else:
        out[y0:y1, x0:x1] = _mirror_tile(img[y0:y1, x1:][:, ::-1], w, 1)[:, ::-1]
    return out


def only_bg_t(img, mask):
    return fill_box_tiled(img, removal_box(mask))


def composite(obj_img, alpha, bg):
    a = alpha[..., None]
    return np.clip(a * obj_img + (1 - a) * bg, 0, 255).astype(np.uint8)


def only_fg(img, alpha):
    return composite(img, alpha, np.broadcast_to(GREY, img.shape).astype(np.float32))


def pairings(meta, seed):
    """Donor index (into meta) for MIXED_SAME and MIXED_RAND. Never self or own duplicate group.
    MIXED_RAND: a different eligible class drawn uniformly, then an image of that class."""
    rng = np.random.default_rng(seed)
    same, rand = [], []
    present = [c for c in CLASSES if (meta.cls.values == c).any()]   # classes failing the mask gate are absent
    by_cls = {c: np.flatnonzero(meta.cls.values == c) for c in present}
    for i, (c, g) in enumerate(zip(meta.cls, meta.group)):
        pool = by_cls[c][meta.group.values[by_cls[c]] != g]
        same.append(rng.choice(pool))
        oc = rng.choice([k for k in present if k != c])
        rand.append(rng.choice(by_cls[oc]))
    return np.array(same), np.array(rand)


def build_test(meta, frames, masks):
    pair = {p: pairings(meta, p) for p in range(N_PAIRINGS)}
    json.dump({p: {"same": meta.id.values[s].tolist(), "rand": meta.id.values[r].tolist()} for p, (s, r) in pair.items()},
              open(VDIR / "test_pairings.json", "w"))
    for kind in ("base", "erode3", "dilate3"):
        mk = np.stack([perturb(m, kind) for m in masks])
        alphas = np.stack([paste_alpha(m) for m in mk])
        bgs = np.stack([only_bg_t(f, m) for f, m in zip(frames, mk)])
        out = {"ORIGINAL": frames, "ONLY_FG": np.stack([only_fg(f, a) for f, a in zip(frames, alphas)]), "ONLY_BG_T": bgs}
        for p, (s, r) in pair.items():
            out[f"MIXED_SAME_{p}"] = np.stack([composite(frames[i], alphas[i], bgs[s[i]]) for i in range(len(frames))])
            out[f"MIXED_RAND_{p}"] = np.stack([composite(frames[i], alphas[i], bgs[r[i]]) for i in range(len(frames))])
        for k, v in out.items():
            if kind != "base" and k == "ORIGINAL":
                continue
            np.save(VDIR / f"test_{kind}_{k}.npy", v)
    return pair


def fixed_box(train_masks):
    """Centred box containing the dilated object of 75% of eligible training images (per axis)."""
    hy, hx = [], []
    for m in train_masks:
        y0, y1, x0, x1 = removal_box(m)
        hy.append(max(FRAME / 2 - y0, y1 - FRAME / 2)); hx.append(max(FRAME / 2 - x0, x1 - FRAME / 2))
    ry, rx = np.percentile(hy, 75), np.percentile(hx, 75)
    ry, rx = min(ry, FRAME / 2 - 16), min(rx, FRAME / 2 - 16)      # keep a >= 16 px strip outside
    c = FRAME / 2
    return int(c - ry), int(np.ceil(c + ry)), int(c - rx), int(np.ceil(c + rx))


def inside(mask, box):
    y0, y1, x0, x1 = removal_box(mask)
    return y0 >= box[0] and y1 <= box[1] and x0 >= box[2] and x1 <= box[3]


def main():
    el = pd.read_csv(WORK / "eligibility.csv")
    frames_all = np.load(WORK / "trashnet_frames.npy", mmap_mode="r")
    sal = np.load(WORK / "u2net_frame_saliency.npy", mmap_mode="r")
    info = {}
    # per-split eligible sets
    sets = {}
    for split in ("train", "val", "test"):
        meta = el[el.eligible & (el.split == split)].reset_index(drop=True)
        frames = np.asarray(frames_all[meta.row.values])
        masks = np.stack([(sal[r].astype(np.float32) >= 0.5).astype(np.uint8) for r in meta.row])
        sets[split] = (meta, frames, masks)
        meta.to_csv(VDIR / f"{split}_eligible_meta.csv", index=False)
        np.save(VDIR / f"{split}_masks.npy", masks)
    # training-regime ingredients
    meta, frames, masks = sets["train"]
    np.save(VDIR / "train_alpha.npy", np.stack([paste_alpha(m) for m in masks]).astype(np.float16))
    # test variants
    build_test(*sets["test"])
    # Probe A: ONLY_BG_T for every split, and the fixed-box control
    box = fixed_box(sets["train"][2])
    info["fixed_box_y0y1x0x1"] = list(map(int, box))
    for split, (meta, frames, masks) in sets.items():
        np.save(VDIR / f"{split}_ONLY_BG_T.npy", np.stack([only_bg_t(f, m) for f, m in zip(frames, masks)]))
        ok = np.array([inside(m, box) for m in masks])
        np.save(VDIR / f"{split}_fixedbox_keep.npy", ok)
        np.save(VDIR / f"{split}_ONLY_BG_FIXED.npy", np.stack([fill_box_tiled(f, box) for f in frames[ok]]))
        info[f"{split}_eligible"] = int(len(meta)); info[f"{split}_fixedbox_kept"] = int(ok.sum())
        info[f"{split}_fixedbox_kept_by_class"] = meta.cls[ok].value_counts().to_dict()
    json.dump(info, open(RESULTS / "variants_info.json", "w"), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
