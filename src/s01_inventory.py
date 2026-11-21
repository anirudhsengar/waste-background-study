"""Stage 1: inventory both datasets, check counts, cache 224-frame images, plot samples.

Outputs (data/work): trashnet_manifest.csv, realwaste_manifest.csv,
trashnet_frames.npy, realwaste_frames.npy; results/counts.csv; figures.
"""
import hashlib

import imagehash
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from .common import (CLASSES, FIGS, REALWASTE_DIR, REALWASTE_MAP, RESULTS, TRASHNET_DIR, WORK,
                     read_rgb, to_frame)

EXPECTED_TN = {"cardboard": 403, "glass": 501, "metal": 410, "paper": 594, "plastic": 482, "trash": 137}
EXPECTED_RW = {"Cardboard": 461, "Food Organics": 411, "Glass": 420, "Metal": 790, "Miscellaneous Trash": 495,
               "Paper": 500, "Plastic": 921, "Textile Trash": 318, "Vegetation": 436}


def build_trashnet():
    rows, frames = [], []
    for c in CLASSES:
        files = sorted((TRASHNET_DIR / c).glob("*.jpg"), key=lambda p: int(p.stem[len(c):]))
        assert len(files) == EXPECTED_TN[c], (c, len(files))
        for p in files:
            img = read_rgb(p)
            pil = Image.fromarray(img)
            rows.append(dict(id=p.stem, cls=c, path=str(p.relative_to(TRASHNET_DIR)), h=img.shape[0], w=img.shape[1],
                             md5=hashlib.md5(img.tobytes()).hexdigest(),
                             phash=str(imagehash.phash(pil)), dhash=str(imagehash.dhash(pil))))
            frames.append(to_frame(img))
    return pd.DataFrame(rows), np.stack(frames)


def build_realwaste():
    rows, frames = [], []
    for folder, n in EXPECTED_RW.items():
        files = sorted((REALWASTE_DIR / folder).glob("*.jpg"))
        assert len(files) == n, (folder, len(files))
        if folder not in REALWASTE_MAP:
            continue
        for p in files:
            img = read_rgb(p)
            pil = Image.fromarray(img)
            rows.append(dict(id=p.stem, folder=folder, cls=REALWASTE_MAP[folder], path=str(p.relative_to(REALWASTE_DIR)),
                             h=img.shape[0], w=img.shape[1], md5=hashlib.md5(img.tobytes()).hexdigest(),
                             phash=str(imagehash.phash(pil)), dhash=str(imagehash.dhash(pil))))
            frames.append(to_frame(img))
    return pd.DataFrame(rows), np.stack(frames)


def sample_grid(df, frames, title, out, per_class=8, seed=0):
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(len(CLASSES), per_class, figsize=(per_class * 1.3, len(CLASSES) * 1.4))
    for r, c in enumerate(CLASSES):
        idx = rng.choice(np.flatnonzero(df.cls.values == c), per_class, replace=False)
        for k, i in enumerate(idx):
            ax = axes[r, k]
            ax.imshow(frames[i]); ax.set_xticks([]); ax.set_yticks([])
            if k == 0:
                ax.set_ylabel(c, fontsize=9)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=130); plt.close(fig)


def main():
    tn, tn_frames = build_trashnet()
    rw, rw_frames = build_realwaste()
    tn.to_csv(WORK / "trashnet_manifest.csv", index=False)
    rw.to_csv(WORK / "realwaste_manifest.csv", index=False)
    np.save(WORK / "trashnet_frames.npy", tn_frames)
    np.save(WORK / "realwaste_frames.npy", rw_frames)

    counts = pd.DataFrame({"TrashNet": tn.cls.value_counts(), "RealWaste (6-class)": rw.cls.value_counts()}).loc[CLASSES]
    counts.loc["total"] = counts.sum()
    counts.to_csv(RESULTS / "counts.csv")
    print(counts)
    print("TrashNet sizes:", tn.groupby(["w", "h"]).size().to_dict(), "RealWaste sizes:", rw.groupby(["w", "h"]).size().to_dict())

    sample_grid(tn, tn_frames, "TrashNet (224 x 224 canonical frame)", FIGS / "samples_trashnet.png")
    sample_grid(rw, rw_frames, "RealWaste, six mapped classes (224 x 224 canonical frame)", FIGS / "samples_realwaste.png")

    fig, ax = plt.subplots(figsize=(6, 3))
    counts.drop("total").plot.bar(ax=ax, color=["#4C72B0", "#DD8452"])
    ax.set_ylabel("images"); ax.set_title("Class counts"); plt.xticks(rotation=0)
    fig.tight_layout(); fig.savefig(FIGS / "class_counts.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
