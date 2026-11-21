"""Stage 5b: six-class edited test images from published polygon masks, training masks, and the corner probe.

Uses the polygon re-annotation of TrashNet (Rohit, Tabassum & Raihan, 2026) for all six classes.
  - Edited test images (ORIGINAL, ONLY_FG, ONLY_BG_T, MIXED_SAME[p], MIXED_RAND[p], p = 0..4) for mask
    versions base / erode3 / dilate3, with the eligibility rules listed in results/variants6_info.json.
  - Background-only probe sets (ONLY_BG_T) for training, validation and test.
  - Corner probe: the four 16 x 16-pixel corners of images whose polygon, dilated by 4 pixels, touches no
    corner (810 train / 182 val / 194 test).
  - Training masks, background-only training images and paste alphas for the paste6 / shuffle6 / rand6 regimes.

Outputs in data/work/variants6/ and results/variants6_info.json; the binary polygon masks are packed into
data/work/poly6_masks_packed.npy.

  python -m src.s05_variants6
"""
from __future__ import annotations

import json

import cv2
import numpy as np
import pandas as pd

from .common import CLASSES, FRAME, GREY, N_PAIRINGS, RESULTS, WORK, mask_to_frame
from .s04_masks import MIN_STRIP, _crop_stats, load_references, strip_thickness
from .s05_variants import _disk, composite, only_bg_t, only_fg, pairings, paste_alpha, perturb

V6DIR = WORK / "variants6"
V6DIR.mkdir(exist_ok=True)

POLY_PACKED = WORK / "poly6_masks_packed.npy"
POLY_META = WORK / "poly6_table.csv"

# Counterfactual & BG-donor eligibility: >=75% of object inside square crop (crop_loss <= 0.25,
# matching the 25% width reduction of 512x384 -> 384x384), 2-75% frame area, >=8px background strip.
CF_AREA_MIN, CF_AREA_MAX, CF_CROP_LOSS_MAX = 0.02, 0.75, 0.25
# Training foreground paste eligibility: any matched polygon with visible background (2-95% frame area).
FG_AREA_MIN, FG_AREA_MAX = 0.02, 0.95
CORNER_SIZE = 16
CORNER_BUFFER = 4


def build_poly_cache() -> tuple[pd.DataFrame, np.ndarray]:
    """Build or load per-split-row human polygon masks in the 224x224 canonical frame."""
    sp = pd.read_csv(WORK / "split.csv")
    if POLY_PACKED.exists() and POLY_META.exists():
        meta = pd.read_csv(POLY_META)
        if len(meta) == len(sp) and (meta.id.values == sp.id.values).all():
            masks = np.unpackbits(np.load(POLY_PACKED), axis=-1, count=FRAME).astype(np.uint8)
            return meta, masks

    man = pd.read_csv(WORK / "trashnet_manifest.csv")
    refs, status = load_references(man)
    masks = np.zeros((len(sp), FRAME, FRAME), dtype=np.uint8)
    rows = []
    cmask = corner_region_mask(CORNER_SIZE)
    buf_kernel = _disk(CORNER_BUFFER)

    for k, r in sp.iterrows():
        _id = r["id"]
        st_name = status.get(_id, "no reference")
        if _id in refs:
            rf = refs[_id]
            _, cl = _crop_stats(rf.astype(bool))
            mf = mask_to_frame(rf)
            masks[k] = mf
            area = float(mf.mean())
            strip = int(strip_thickness(mf)) if mf.sum() > 0 else 0
            dil = cv2.dilate(mf, buf_kernel).astype(bool)
            corners_clean = bool((mf.sum() > 0) and not (dil & cmask).any())
        else:
            cl, area, strip, corners_clean = 1.0, 0.0, 0, False
        rows.append(
            dict(
                id=_id,
                poly_status=st_name,
                poly_area=area,
                poly_crop_loss=float(cl),
                poly_strip=strip,
                cf_eligible=bool(
                    (_id in refs)
                    and (CF_AREA_MIN <= area <= CF_AREA_MAX)
                    and (cl <= CF_CROP_LOSS_MAX)
                    and (strip >= MIN_STRIP)
                ),
                fg_paste_eligible=bool((_id in refs) and (FG_AREA_MIN <= area <= FG_AREA_MAX)),
                corner_eligible=corners_clean,
            )
        )

    meta = sp.merge(pd.DataFrame(rows), on="id", how="left")
    meta.to_csv(POLY_META, index=False)
    np.save(POLY_PACKED, np.packbits(masks, axis=-1))
    return meta, masks


def corner_region_mask(csz: int = CORNER_SIZE) -> np.ndarray:
    m = np.zeros((FRAME, FRAME), dtype=bool)
    m[:csz, :csz] = True
    m[:csz, -csz:] = True
    m[-csz:, :csz] = True
    m[-csz:, -csz:] = True
    return m


def make_corner_image(img: np.ndarray, cmask: np.ndarray) -> np.ndarray:
    out = np.broadcast_to(GREY, img.shape).copy()
    out[cmask] = img[cmask]
    return out


def build_test6(meta: pd.DataFrame, frames: np.ndarray, masks: np.ndarray) -> dict:
    pair = {p: pairings(meta, p) for p in range(N_PAIRINGS)}
    json.dump(
        {p: {"same": meta.id.values[s].tolist(), "rand": meta.id.values[r].tolist()} for p, (s, r) in pair.items()},
        open(V6DIR / "test_pairings.json", "w"),
    )
    for kind in ("base", "erode3", "dilate3"):
        mk = np.stack([perturb(m, kind) for m in masks])
        alphas = np.stack([paste_alpha(m) for m in mk])
        bgs = np.stack([only_bg_t(f, m) for f, m in zip(frames, mk)])
        out = {
            "ORIGINAL": frames,
            "ONLY_FG": np.stack([only_fg(f, a) for f, a in zip(frames, alphas)]),
            "ONLY_BG_T": bgs,
        }
        for p, (s, r) in pair.items():
            out[f"MIXED_SAME_{p}"] = np.stack([composite(frames[i], alphas[i], bgs[s[i]]) for i in range(len(frames))])
            out[f"MIXED_RAND_{p}"] = np.stack([composite(frames[i], alphas[i], bgs[r[i]]) for i in range(len(frames))])
        for k, v in out.items():
            if kind != "base" and k == "ORIGINAL":
                continue
            np.save(V6DIR / f"test_{kind}_{k}.npy", v)
    return pair


def main() -> None:
    meta_all, masks_all = build_poly_cache()
    frames_all = np.load(WORK / "trashnet_frames.npy", mmap_mode="r")
    cmask = corner_region_mask(CORNER_SIZE)
    info: dict = {
        "cf_criteria": {
            "area_min": CF_AREA_MIN,
            "area_max": CF_AREA_MAX,
            "crop_loss_max": CF_CROP_LOSS_MAX,
            "min_strip_px": MIN_STRIP,
        },
        "fg_paste_criteria": {"area_min": FG_AREA_MIN, "area_max": FG_AREA_MAX},
        "corner_probe": {"corner_size_px": CORNER_SIZE, "dilation_buffer_px": CORNER_BUFFER},
    }

    # 1. Six-class counterfactual & BG-donor sets (train / val / test)
    cf_sets = {}
    for split in ("train", "val", "test"):
        idx = np.flatnonzero(meta_all.cf_eligible & (meta_all.split == split))
        sub = meta_all.iloc[idx].reset_index(drop=True)
        fr = np.asarray(frames_all[sub.row.values])
        mk = masks_all[idx]
        cf_sets[split] = (sub, fr, mk)
        sub.to_csv(V6DIR / f"{split}_cf_meta.csv", index=False)
        np.save(V6DIR / f"{split}_cf_masks.npy", mk)
        np.save(V6DIR / f"{split}_ONLY_BG_T.npy", np.stack([only_bg_t(f, m) for f, m in zip(fr, mk)]))
        info[f"{split}_cf_eligible"] = int(len(sub))
        info[f"{split}_cf_by_class"] = {c: int((sub.cls == c).sum()) for c in CLASSES}

    # Build counterfactual test variants (base, erode3, dilate3)
    build_test6(*cf_sets["test"])

    # 2. Six-class foreground-paste training ingredients (1,519 images across all 6 classes)
    fg_idx = np.flatnonzero(meta_all.fg_paste_eligible & (meta_all.split == "train"))
    fg_sub = meta_all.iloc[fg_idx].reset_index(drop=True)
    fg_masks = masks_all[fg_idx]
    fg_sub.to_csv(V6DIR / "train_fg_meta.csv", index=False)
    np.save(V6DIR / "train_fg_alpha.npy", np.stack([paste_alpha(m) for m in fg_masks]).astype(np.float16))
    info["train_fg_paste_eligible"] = int(len(fg_sub))
    info["train_fg_paste_by_class"] = {c: int((fg_sub.cls == c).sum()) for c in CLASSES}

    # 3. Six-class 16x16 Corner-Patch Probe (untouched pixels, zero box-size cue, zero seams)
    for split in ("train", "val", "test"):
        c_idx = np.flatnonzero(meta_all.corner_eligible & (meta_all.split == split))
        c_sub = meta_all.iloc[c_idx].reset_index(drop=True)
        c_fr = np.asarray(frames_all[c_sub.row.values])
        c_imgs = np.stack([make_corner_image(f, cmask) for f in c_fr])
        c_sub.to_csv(V6DIR / f"{split}_corner_meta.csv", index=False)
        np.save(V6DIR / f"{split}_ONLY_BG_CORNERS.npy", c_imgs)
        info[f"{split}_corner_kept"] = int(len(c_sub))
        info[f"{split}_corner_by_class"] = {c: int((c_sub.cls == c).sum()) for c in CLASSES}

    # Eligibility summary CSV for tables
    tab_cf = pd.crosstab(meta_all.cls, meta_all.split, values=meta_all.cf_eligible, aggfunc="sum")[["train", "val", "test"]]
    tab_tot = pd.crosstab(meta_all.cls, meta_all.split)[["train", "val", "test"]]
    (tab_cf.astype(int).astype(str) + " / " + tab_tot.astype(str)).to_csv(RESULTS / "eligibility6_counts.csv")

    json.dump(info, open(RESULTS / "variants6_info.json", "w"), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
