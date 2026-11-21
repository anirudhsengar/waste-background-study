"""Stage 4: object masks, quality against reference polygons, eligibility.

  python -m src.s04_masks u2net      # U2-Net saliency for every TrashNet image (slow; CPU)
  python -m src.s04_masks evaluate   # Otsu baseline, reference IoU, method choice, eligibility
  python -m src.s04_masks sheets     # blind contact sheets of eligible test masks for the visual check
  python -m src.s04_masks finalize   # applies configs/mask_visual_check.csv -> eligibility.csv

The recorded verdicts in configs/mask_visual_check.csv were made by an AI model (Claude), blind to the class
label; the `rater` column says so.
"""
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import pandas as pd
from PIL import Image

from .common import (CLASSES, FIGS, FRAME, RESULTS, ROOT, SEG_DIR, TRASHNET_DIR, U2NET_ONNX, WORK, crop_box,
                     mask_to_frame, read_rgb, to_frame)

AREA_MIN, AREA_MAX, CROP_LOSS_MAX = 0.02, 0.60, 0.10
MIN_STRIP = 8              # px; ONLY-BG-T needs a background strip at least this thick
VISUAL_CHECK = ROOT / "configs" / "mask_visual_check.csv"


# ---------- U2-Net -------------------------------------------------------------------------
def u2net_saliency(sess, img):
    """rembg-equivalent preprocessing: 320x320 Lanczos, scale by max, ImageNet normalisation."""
    h, w = img.shape[:2]
    x = np.asarray(Image.fromarray(img).resize((320, 320), Image.LANCZOS), dtype=np.float32)
    x = x / max(x.max(), 1e-6)
    x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    pred = sess.run(None, {sess.get_inputs()[0].name: x.transpose(2, 0, 1)[None].astype(np.float32)})[0][0, 0]
    pred = (pred - pred.min()) / max(pred.max() - pred.min(), 1e-6)
    return np.asarray(Image.fromarray((pred * 255).astype(np.uint8)).resize((w, h), Image.LANCZOS), dtype=np.float32) / 255.0


def run_u2net():
    df = pd.read_csv(WORK / "trashnet_manifest.csv")
    opts = ort.SessionOptions(); opts.intra_op_num_threads = 10
    sess = ort.InferenceSession(str(U2NET_ONNX), opts, providers=["CPUExecutionProvider"])
    frame_sal = np.zeros((len(df), FRAME, FRAME), dtype=np.float16)
    full_stats = []
    for i, p in enumerate(df.path):
        img = read_rgb(TRASHNET_DIR / p)
        sal = u2net_saliency(sess, img)
        full_stats.append(_crop_stats(sal >= 0.5))
        frame_sal[i] = to_frame(sal, interp=cv2.INTER_AREA)
        if i % 250 == 0:
            print(i, flush=True)
    np.save(WORK / "u2net_frame_saliency.npy", frame_sal)
    pd.DataFrame(full_stats, columns=["u2net_full_area", "u2net_crop_loss"]).assign(id=df.id).to_csv(WORK / "u2net_full_stats.csv", index=False)


def _crop_stats(full_mask):
    """(object area fraction in the full image, fraction of the object lost to the centre crop)."""
    y0, y1, x0, x1 = crop_box(*full_mask.shape)
    tot = full_mask.sum()
    kept = full_mask[y0:y1, x0:x1].sum()
    return full_mask.mean(), (1 - kept / tot) if tot else 0.0


# ---------- Otsu (taught baseline) --------------------------------------------------------
def otsu_mask(img):
    """Object = darker than the board (Otsu on grey) OR more saturated (Otsu on S); cleaned, holes filled."""
    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sat = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[..., 1]
    grey = cv2.GaussianBlur(grey, (5, 5), 0); sat = cv2.GaussianBlur(sat, (5, 5), 0)
    _, g = cv2.threshold(grey, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, s = cv2.threshold(sat, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = (g | s).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(m)
    cv2.drawContours(filled, cnts, -1, 1, thickness=cv2.FILLED)
    return filled


# ---------- reference polygons --------------------------------------------------------------
def load_references(df):
    """Rasterised Roboflow/Mendeley polygons matched by file name, in 512 x 384 geometry.

    Excluded: no file, class index disagreeing with the TrashNet folder (relabelled), or a
    polygon that is the whole frame (no usable outline)."""
    refs, status = {}, {}
    files = {}
    for f in SEG_DIR.glob("*/labels/*.txt"):
        files.setdefault(f.name.split("_jpg")[0], f)
    for _id, cls in zip(df.id, df.cls):
        f = files.get(_id)
        if f is None:
            status[_id] = "no reference"; continue
        lines = [l.split() for l in f.read_text().strip().splitlines() if l.strip()]
        if any(CLASSES[int(l[0])] != cls for l in lines):
            status[_id] = "label mismatch"; continue
        m = np.zeros((384, 512), np.uint8)
        for l in lines:
            pts = (np.array(l[1:], float).reshape(-1, 2) * [512, 384]).round().astype(np.int32)
            cv2.fillPoly(m, [pts], 1)
        if m.mean() > 0.97:
            status[_id] = "whole-frame polygon"; continue
        refs[_id] = m; status[_id] = "matched"
    return refs, status


def iou_parts(pred, ref):
    inter = (pred & ref).sum(); union = (pred | ref).sum()
    return (inter / union if union else 1.0,
            1 - inter / ref.sum() if ref.sum() else 0.0,            # missed-object fraction
            (pred & ~ref).sum() / max((~ref).sum(), 1))              # background-leak fraction


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x); rng = np.random.default_rng(seed)
    b = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return np.percentile(b, [2.5, 97.5])


def strip_thickness(mask_frame, dil=8):
    """Largest background strip outside the dilated object box (px); see variants.only_bg_t."""
    d = cv2.dilate(mask_frame, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dil + 1,) * 2))
    ys, xs = np.nonzero(d)
    if len(ys) == 0:
        return 0
    return max(ys.min(), FRAME - 1 - ys.max(), xs.min(), FRAME - 1 - xs.max())


def evaluate():
    df = pd.read_csv(WORK / "split.csv")
    man = pd.read_csv(WORK / "trashnet_manifest.csv")
    sal = np.load(WORK / "u2net_frame_saliency.npy")[df.row.values].astype(np.float32)
    full = pd.read_csv(WORK / "u2net_full_stats.csv").set_index("id").loc[df.id]
    refs, status = load_references(man)
    df["ref_status"] = df.id.map(status)
    rows = []
    otsu_frames = np.zeros((len(df), FRAME, FRAME), np.uint8)
    for k, (_id, path) in enumerate(zip(df.id, man.set_index("id").loc[df.id].path)):
        img = read_rgb(TRASHNET_DIR / path)
        om = otsu_mask(img)
        otsu_frames[k] = mask_to_frame(om)
        u = (sal[k] >= 0.5).astype(np.uint8)
        r = dict(id=_id)
        if _id in refs:
            ref_full = refs[_id]
            r["ref_area_full"], r["ref_crop_loss"] = _crop_stats(ref_full.astype(bool))
            ref = mask_to_frame(ref_full).astype(bool)
            r["u2net_iou"], r["u2net_missed"], r["u2net_leak"] = iou_parts(u.astype(bool), ref)
            r["otsu_iou"], r["otsu_missed"], r["otsu_leak"] = iou_parts(otsu_frames[k].astype(bool), ref)
        rows.append(r)
    q = pd.DataFrame(rows)
    df = df.merge(q, on="id", how="left")
    np.save(WORK / "otsu_frame_masks.npy", otsu_frames)

    # method choice and IoU gate on images with a matched reference
    ref_ok = df[df.ref_status == "matched"]
    summ = []
    for m in ("u2net", "otsu"):
        for c in CLASSES + ["all"]:
            s = ref_ok if c == "all" else ref_ok[ref_ok.cls == c]
            lo, hi = boot_ci(s[f"{m}_iou"])
            summ.append(dict(method=m, cls=c, n=len(s), mean_iou=s[f"{m}_iou"].mean(), ci_lo=lo, ci_hi=hi,
                             median_iou=s[f"{m}_iou"].median(), missed=s[f"{m}_missed"].mean(), leak=s[f"{m}_leak"].mean()))
    summ = pd.DataFrame(summ)
    summ.to_csv(RESULTS / "mask_quality.csv", index=False)
    print(summ.round(3).to_string())
    print(df.ref_status.value_counts())

    # eligibility (U2-Net chosen if it wins; gate reported)
    df["area"] = sal.__ge__(0.5).mean((1, 2))
    df["crop_loss"] = full.u2net_crop_loss.values
    df["strip"] = [strip_thickness((s >= 0.5).astype(np.uint8)) for s in sal]
    df["auto_eligible"] = df.area.between(AREA_MIN, AREA_MAX) & (df.crop_loss <= CROP_LOSS_MAX) & (df.strip >= MIN_STRIP)
    df.drop(columns=[c for c in df.columns if c.startswith("otsu_")]).to_csv(WORK / "mask_table.csv", index=False)
    print(pd.crosstab(df.cls, [df.split, df.auto_eligible]))

    # crop loss per class from reference polygons
    cl = ref_ok.assign(rcl=df.loc[ref_ok.index, "ref_crop_loss"]).groupby("cls").rcl.describe(percentiles=[.5, .9])
    cl["share_over_10pct"] = ref_ok.assign(rcl=df.loc[ref_ok.index, "ref_crop_loss"]).groupby("cls").rcl.apply(lambda s: (s > 0.10).mean())
    cl.to_csv(RESULTS / "crop_loss_reference.csv")
    _figure(df, sal, otsu_frames, refs)
    _area_hist(df)


def _figure(df, sal, otsu_frames, refs):
    rng = np.random.default_rng(1)
    frames = np.load(WORK / "trashnet_frames.npy", mmap_mode="r")
    fig, ax = plt.subplots(4, 6, figsize=(11, 7.6))
    for j, c in enumerate(CLASSES):
        k = rng.choice(np.flatnonzero((df.cls == c) & (df.ref_status == "matched")))
        img = frames[df.row[k]]
        ref = mask_to_frame(refs[df.id[k]])
        for i, (m, t) in enumerate([(None, c), (ref, "reference"), (otsu_frames[k], "Otsu"), ((sal[k] >= 0.5).astype(np.uint8), "U²-Net")]):
            a = ax[i, j]
            if m is None:
                a.imshow(img)
            else:
                ov = img.copy(); ov[m.astype(bool)] = (0.5 * ov[m.astype(bool)] + [0, 110, 0]).astype(np.uint8)
                a.imshow(ov)
            a.set_xticks([]); a.set_yticks([])
            if j == 0: a.set_ylabel(t if i else "image")
            if i == 0: a.set_title(c)
    fig.tight_layout(); fig.savefig(FIGS / "masks_otsu_vs_u2net.png", dpi=110); plt.close(fig)


def _area_hist(df):
    fig, ax = plt.subplots(figsize=(6, 3))
    for c in CLASSES:
        ax.hist(df.area[df.cls == c], bins=np.linspace(0, 1, 41), histtype="step", label=c)
    ax.axvspan(AREA_MIN, AREA_MAX, color="grey", alpha=0.12, label="eligible area")
    ax.set_xlabel("object area fraction (U²-Net, 224 frame)"); ax.set_ylabel("images"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIGS / "object_area_trashnet.png", dpi=150); plt.close(fig)


# ---------- blind visual check ----------------------------------------------------------------
def sheets():
    df = pd.read_csv(WORK / "mask_table.csv")
    sal = np.load(WORK / "u2net_frame_saliency.npy")
    frames = np.load(WORK / "trashnet_frames.npy", mmap_mode="r")
    cand = df[df.auto_eligible & (df.split == "test")].sample(frac=1, random_state=0).reset_index(drop=True)
    out = WORK / "mask_check"; out.mkdir(exist_ok=True)
    cand[["id"]].assign(sheet_pos=range(len(cand))).to_csv(out / "order.csv", index=False)
    per = 30
    for s in range(0, len(cand), per):
        sub = cand.iloc[s:s + per]
        fig, axes = plt.subplots(5, 6, figsize=(12, 10.5))
        for a in axes.flat: a.axis("off")
        for k, (row, _) in enumerate(zip(sub.row, sub.id)):
            m = sal[row].astype(np.float32) >= 0.5
            img = np.array(frames[row])
            edge = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(bool)
            img[edge] = [255, 0, 255]
            axes.flat[k].imshow(img); axes.flat[k].set_title(str(s + k), fontsize=9)
        fig.tight_layout(); fig.savefig(out / f"sheet_{s:03d}.png", dpi=70); plt.close(fig)
    print(len(cand), "test images to check")


def finalize():
    df = pd.read_csv(WORK / "mask_table.csv")
    chk = pd.read_csv(VISUAL_CHECK)
    order = pd.read_csv(WORK / "mask_check" / "order.csv").merge(chk, on="sheet_pos")
    bad = set(order.id[order.verdict == "fail"])
    # training images: no visual check; reference agreement used where a reference exists
    # pre-registered IoU gate: U2-Net must beat Otsu; classes below 0.80 mean IoU leave the background tests
    q = pd.read_csv(RESULTS / "mask_quality.csv")
    allm = q[q.cls == "all"].set_index("method").mean_iou
    assert allm["u2net"] >= allm["otsu"], "Otsu beats U2-Net: switch the mask method before continuing"
    uq = q[(q.method == "u2net") & (q.cls != "all")].set_index("cls").mean_iou
    failed = sorted(uq[uq < 0.80].index)
    gate = dict(overall_mean_iou=float(allm["u2net"]), overall_pass=bool(allm["u2net"] >= 0.85), failed_classes=failed)
    pd.Series(gate).to_json(RESULTS / "mask_gate.json")
    print("IoU gate:", gate)
    df["visual_pass"] = np.where(df.split == "test", ~df.id.isin(bad),
                                 ~((df.ref_status == "matched") & (df.u2net_iou < 0.5)))
    df["eligible"] = df.auto_eligible & df.visual_pass & ~df.cls.isin(failed)
    df.to_csv(WORK / "eligibility.csv", index=False)
    tab = pd.crosstab(df.cls, df.split, values=df.eligible, aggfunc="sum")[["train", "val", "test"]]
    tot = pd.crosstab(df.cls, df.split)[["train", "val", "test"]]
    out = tab.astype(int).astype(str) + " / " + tot.astype(str)
    out.to_csv(RESULTS / "eligibility_counts.csv")
    print(out); print("visual-check failures (test):", len(bad))


if __name__ == "__main__":
    {"u2net": run_u2net, "evaluate": evaluate, "sheets": sheets, "finalize": finalize}[sys.argv[1]]()
