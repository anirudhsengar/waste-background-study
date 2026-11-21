"""Stage 11: the three paper figures (400 DPI, serif, drawn at the 160 mm printed text width).

  - results/figures/icasf_fig1_protocol.png  : edited test images, corner probe, a T-rand training image, RealWaste
  - results/figures/icasf_fig2_shortcuts.png : (a) background-only probes, (b) accuracy per edited image, (c) flip rate
  - results/figures/icasf_fig3_transfer.png  : (a) RealWaste macro-F1 by regime and backbone, (b)-(c) confusion matrices

Figs 2-3 read only results/ tables. Fig 1 needs the prepared image arrays (data preparation, s01-s05); it is
skipped with a message when they are missing.

  python -m src.s11_figures            # all figures whose inputs exist
  python -m src.s11_figures fig2 fig3  # selected figures
"""
from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import FIGS, RESULTS, WORK
from .s05_variants6 import V6DIR

# Times New Roman style to match the paper
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    # Figures 1-3 are drawn at their printed size (PRINT_W inches wide), so these are true printed point sizes.
    "font.size": 7.5,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
})
PRINT_W = 6.3  # inches: the 160 mm printed text width; Figs 1-3 are placed in the paper at this width
LETTER = dict(fontsize=9, fontweight="bold")
NET_ABBR = {"M1a colour SVM": "Colour", "M1b SIFT SVM": "SIFT", "M2 ResNet-18": "RN-PF",
            "M2b ResNet-18 (full-FT)": "RN-FT", "M3 MobileNetV3-L": "MNv3-L", "M4 Swin-T": "Swin-T"}

# Colorblind-safe Okabe-Ito academic palette
PAL = {
    "blue": "#0072B2",
    "vermilion": "#D55E00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "amber": "#E69F00",
    "sky": "#56B4E9",
    "grey": "#666666",
}


def fig1_protocol() -> None:
    """Fig. 1: edited test images, the corner probe, a random-texture training image and a RealWaste image."""
    tm6 = pd.read_csv(V6DIR / "test_cf_meta.csv")
    cm6 = pd.read_csv(V6DIR / "test_corner_meta.csv")
    corner_ids = {cid: pos for pos, cid in enumerate(cm6.id.values)}
    v = {
        k: np.load(V6DIR / f"test_base_{k}.npy", mmap_mode="r")
        for k in ("ORIGINAL", "ONLY_FG", "ONLY_BG_T", "MIXED_SAME_0", "MIXED_RAND_0")
    }
    corners = np.load(V6DIR / "test_ONLY_BG_CORNERS.npy", mmap_mode="r")
    masks6 = np.load(V6DIR / "test_cf_masks.npy", mmap_mode="r")
    rw_meta = pd.read_csv(WORK / "realwaste_manifest.csv")
    rw_frames = np.load(WORK / "realwaste_frames.npy", mmap_mode="r")

    from .s05_variants import composite, paste_alpha

    show_classes = ["glass", "plastic"]
    safe_dtd_seeds = {"cardboard": 11, "glass": 27, "plastic": 68}
    rows = []
    for c in show_classes:
        cands = [i for i in np.flatnonzero(tm6.cls.values == c) if tm6.id.values[i] in corner_ids]
        rows.append(cands[min(1, len(cands) - 1)])

    col_titles = ["Original", "Object\non grey", "Background\nonly", "Same-class\nbackground",
                  "Other-class\nbackground", "Corner\nprobe", "Random-texture\ntraining", "RealWaste\n(same class)"]
    fig, axes = plt.subplots(len(show_classes), len(col_titles), figsize=(PRINT_W, 1.8))
    dtd_all = np.load(WORK / "dtd_train_256.npy", mmap_mode="r")
    rw_rng = np.random.default_rng(7)
    for r, idx in enumerate(rows):
        cls_name = show_classes[r]
        cid = tm6.id.values[idx]
        alpha = paste_alpha(masks6[idx])
        dtd_bg = np.asarray(dtd_all[safe_dtd_seeds[cls_name] * 17 % len(dtd_all)][16:240, 16:240], dtype=np.float32)
        trand_img = composite(v["ORIGINAL"][idx].astype(np.float32), alpha, dtd_bg)
        rw_idx = int(rw_rng.choice(np.flatnonzero(rw_meta.cls.values == cls_name)))
        imgs = [v["ORIGINAL"][idx], v["ONLY_FG"][idx], v["ONLY_BG_T"][idx], v["MIXED_SAME_0"][idx],
                v["MIXED_RAND_0"][idx], corners[corner_ids[cid]], trand_img, rw_frames[rw_idx]]
        for c, im in enumerate(imgs):
            ax = axes[r, c]
            ax.imshow(im)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_linewidth(0.4)
                sp.set_edgecolor("#bbbbbb")
            if r == 0:
                ax.set_title(col_titles[c], fontsize=7, pad=3)
            if c == 0:
                ax.set_ylabel(cls_name.capitalize(), fontsize=7.5)
    fig.subplots_adjust(left=0.035, right=0.995, top=0.83, bottom=0.01, wspace=0.05, hspace=0.06)
    fig.savefig(FIGS / "icasf_fig1_protocol.png", dpi=400)
    plt.close(fig)


BIG = {"fontsize": 8.5}          # Figs 2-3 axis labels (printed pt)
TICK = 8                          # Figs 2-3 tick labels (printed pt)
LETTER10 = dict(fontsize=10, fontweight="bold")


def _style(ax, yticks):
    ax.set_yticks(yticks)
    ax.tick_params(labelsize=TICK, width=0.7, length=3)
    ax.yaxis.grid(True, linestyle=":", lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def fig2_shortcuts() -> None:
    """Fig. 2 (printed 6.3 x 2.6 in): (a) background-only probes, (b) accuracy on each edited test image,
    (c) flip rate, all under T-orig."""
    pr = pd.read_csv(RESULTS / "probes.csv")
    rel = pd.read_csv(RESULTS / "reliance6.csv")
    chance = 1.0 / 6.0
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(PRINT_W, 2.5),
                                           gridspec_kw={"width_ratios": [1.1, 2.55, 1.2]})
    # (a) probes
    model_order = [m for m in ["M1a colour SVM", "M1b SIFT SVM", "M2 ResNet-18", "M2b ResNet-18 (full-FT)"]
                   if m in set(pr[pr.probe == "bg6"].model)]
    x = np.arange(len(model_order))
    bw = 0.4
    for k_idx, (pkind, lbl, col) in enumerate([("bg6", "Bg only", PAL["amber"]), ("corners", "Corners", PAL["green"])]):
        vals, lo_e, hi_e = [], [], []
        for m in model_order:
            sub = pr[(pr.model == m) & (pr.probe == pkind) & (pr.masks == "base") &
                     (pr.rep == ("0" if "SVM" in m else "ens"))]
            r = sub.iloc[0]
            vals.append(r.balanced_acc)
            lo_e.append(r.balanced_acc - r.ci_lo)
            hi_e.append(r.ci_hi - r.balanced_acc)
        ax_a.bar(x + (k_idx - 0.5) * bw, vals, bw * 0.95, yerr=[lo_e, hi_e], capsize=2, color=col, label=lbl,
                 error_kw={"lw": 0.8})
    ax_a.axhline(chance, ls="--", color="#222222", lw=0.9)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([NET_ABBR[m] for m in model_order], rotation=45, ha="right", rotation_mode="anchor")
    ax_a.set_ylim(0, 1.0)
    ax_a.set_ylabel("Balanced accuracy", **BIG)
    ax_a.legend(frameon=False, loc="upper left", fontsize=8, handlelength=1.0, borderaxespad=0.1,
                labelspacing=0.2, handletextpad=0.4)
    _style(ax_a, [0, 0.5, 1.0])
    ax_a.text(-0.42, 1.04, "(a)", transform=ax_a.transAxes, **LETTER10)

    # (b) accuracy on each edited test image
    cols = {"M1a colour SVM": PAL["vermilion"], "M1b SIFT SVM": PAL["amber"], "M2 ResNet-18": PAL["blue"],
            "M2b ResNet-18 (full-FT)": PAL["green"], "M3 MobileNetV3-L": PAL["sky"], "M4 Swin-T": PAL["purple"]}
    base = rel[(rel.regime == "T-orig") & (rel.masks == "base")]
    rel_models = [m for m in cols if m in set(base.model)]
    vars_ = ["ORIGINAL", "ONLY_FG", "ONLY_BG_T", "MIXED_SAME", "MIXED_RAND"]
    vx = np.arange(len(vars_)) * 1.25  # wider spacing between groups
    vw = 0.95 / len(rel_models)
    rowof = lambda m: base[(base.model == m) & (base.rep == ("0" if "SVM" in m else "ens"))].iloc[0]
    for j, m in enumerate(rel_models):
        r = rowof(m)
        vals = [r[f"acc_{v}"] for v in vars_]
        err = [[r[f"acc_{v}"] - r[f"acc_{v}_lo"] for v in vars_], [r[f"acc_{v}_hi"] - r[f"acc_{v}"] for v in vars_]]
        ax_b.bar(vx + (j - (len(rel_models) - 1) / 2.0) * vw, vals, vw * 0.95, yerr=err, capsize=1.0,
                 color=cols[m], label=NET_ABBR[m], error_kw={"lw": 0.6})
    ax_b.axhline(chance, ls="--", color="#222222", lw=0.9)
    ax_b.set_xticks(vx)
    ax_b.set_xticklabels(["Original", "On grey", "Bg only", "Same bg", "Other bg"])
    ax_b.set_ylim(0, 1.0)
    ax_b.set_ylabel("Accuracy", **BIG)
    _style(ax_b, [0, 0.5, 1.0])
    ax_b.text(-0.13, 1.04, "(b)", transform=ax_b.transAxes, **LETTER10)
    fig.legend(*ax_b.get_legend_handles_labels(), loc="upper center", ncol=len(rel_models), frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.54, 1.0), handlelength=1.1, columnspacing=1.0, handletextpad=0.35)

    # (c) flip rate
    fx = np.arange(len(rel_models))
    rr = [rowof(m) for m in rel_models]
    ax_c.bar(fx, [r.flip_rate for r in rr], 0.7,
             yerr=[[r.flip_rate - r.flip_rate_lo for r in rr], [r.flip_rate_hi - r.flip_rate for r in rr]],
             capsize=2, color=[cols[m] for m in rel_models], error_kw={"lw": 0.8})
    ax_c.set_xticks(fx)
    ax_c.set_xticklabels([NET_ABBR[m] for m in rel_models], rotation=50, ha="right", rotation_mode="anchor")
    ax_c.set_ylim(0, 0.8)
    ax_c.set_ylabel("Flip rate", **BIG)
    _style(ax_c, [0, 0.4, 0.8])
    ax_c.text(-0.42, 1.04, "(c)", transform=ax_c.transAxes, **LETTER10)
    fig.subplots_adjust(left=0.085, right=0.995, top=0.85, bottom=0.2, wspace=0.42)
    fig.savefig(FIGS / "icasf_fig2_shortcuts.png", dpi=400)
    plt.close(fig)


def fig3_transfer() -> None:
    """Fig. 3 (printed 6.3 x 2.6 in): (a) RealWaste macro-F1 by regime and network; (b, c) RealWaste confusion
    matrices (row %) of the partly frozen ResNet-18 (the primary model, as in the text) under T-orig and T-rand."""
    perf = pd.read_csv(RESULTS / "performance.csv")
    cms_raw = np.load(RESULTS / "confusion_matrices.npy", allow_pickle=True).item()
    W, H = PRINT_W, 2.5
    cm_s = 1.55                                   # matrix side (in): cells of ~0.26 in for 8 pt numbers
    b_y, gap, lab_w = 0.47, 0.05, 0.42            # bottom margin, gap between matrices, room for class labels
    c_x = W - 0.02 - cm_s
    b_x = c_x - gap - cm_s
    fig = plt.figure(figsize=(W, H))
    ax_a = fig.add_axes([0.48 / W, b_y / H, (b_x - lab_w - 0.48) / W, (H - b_y - 0.3) / H])
    ax_b = fig.add_axes([b_x / W, b_y / H, cm_s / W, cm_s / H])
    ax_c = fig.add_axes([c_x / W, b_y / H, cm_s / W, cm_s / H])

    rw = perf[perf.set == "rw"]
    neural = [m for m in ["M2 ResNet-18", "M2b ResNet-18 (full-FT)", "M3 MobileNetV3-L", "M4 Swin-T"]
              if len(rw[(rw.model == m) & (rw.regime == "T-rand")])]
    regs = [r for r in ["T-orig", "T-jitter", "T-paste", "T-shuffle", "T-rand"]
            if len(rw[(rw.model == neural[0]) & (rw.regime == r) & (rw.rep == "ens")])]
    rcol = {"T-orig": PAL["grey"], "T-jitter": PAL["amber"], "T-paste": PAL["vermilion"],
            "T-shuffle": PAL["purple"], "T-rand": PAL["blue"]}
    gx = np.arange(len(neural)) * 1.15
    bw = 0.95 / len(regs)
    for r_i, reg in enumerate(regs):
        off = (r_i - (len(regs) - 1) / 2.0) * bw
        vals, lo_e, hi_e = [], [], []
        for m_i, m in enumerate(neural):
            e = rw[(rw.model == m) & (rw.regime == reg) & (rw.rep == "ens")].iloc[0]
            vals.append(e.macro_f1)
            lo_e.append(e.macro_f1 - e.macro_f1_lo)
            hi_e.append(e.macro_f1_hi - e.macro_f1)
            seeds = rw[(rw.model == m) & (rw.regime == reg) & rw.rep.isin(["0", "1", "2"])]
            ax_a.scatter(np.full(len(seeds), gx[m_i] + off), seeds.macro_f1, c="#111111", s=5, zorder=4, lw=0)
        ax_a.bar(gx + off, vals, bw * 0.92, yerr=[lo_e, hi_e], capsize=1.0, color=rcol[reg], label=reg,
                 error_kw={"lw": 0.6})
    ax_a.set_xticks(gx)
    ax_a.set_xticklabels([NET_ABBR[m] for m in neural])
    ax_a.set_ylim(0, 0.8)
    ax_a.set_ylabel("RealWaste macro-F1", **BIG)
    _style(ax_a, [0, 0.4, 0.8])
    ax_a.legend(frameon=False, ncol=3, loc="upper left", fontsize=8, handlelength=1.0, columnspacing=0.8,
                borderaxespad=0.05, labelspacing=0.2, handletextpad=0.35)
    ax_a.text(-0.2, 1.03, "(a)", transform=ax_a.transAxes, **LETTER10)

    cls_abbr = ["Card", "Glass", "Metal", "Paper", "Plast", "Trash"]
    for tag, ax, reg in [("(b)", ax_b, "T-orig"), ("(c)", ax_c, "T-rand")]:
        cm = cms_raw[f"M2 ResNet-18|{reg}|ens|rw"]
        pctm = 100 * cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(pctm, cmap="Blues", vmin=0, vmax=75, aspect="auto")
        for i in range(6):
            for j in range(6):
                ax.text(j, i, f"{pctm[i, j]:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if pctm[i, j] > 40 else "#111111")
        ax.set_xticks(range(6))
        ax.set_yticks(range(6))
        ax.set_xticklabels(cls_abbr, rotation=90, fontsize=TICK)
        ax.set_yticklabels(cls_abbr if tag == "(b)" else [], fontsize=TICK)
        ax.tick_params(length=0, pad=2)
        ax.set_title(f"{tag} {reg}", fontsize=9, pad=3)
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(0.6)
    fig.savefig(FIGS / "icasf_fig3_transfer.png", dpi=400)
    plt.close(fig)


FIG1_INPUTS = [V6DIR / "test_base_ORIGINAL.npy", V6DIR / "test_cf_masks.npy", V6DIR / "test_ONLY_BG_CORNERS.npy",
               WORK / "realwaste_frames.npy", WORK / "dtd_train_256.npy"]
FIGURES = {"fig1": fig1_protocol, "fig2": fig2_shortcuts, "fig3": fig3_transfer}


def build_all_publication_figures(names=None) -> None:
    for name in names or list(FIGURES):
        if name == "fig1" and not all(p.exists() for p in FIG1_INPUTS):
            print("skip fig1: prepared image arrays missing (run data preparation first)")
            continue
        FIGURES[name]()
        print("wrote", name)


if __name__ == "__main__":
    build_all_publication_figures(sys.argv[1:])
