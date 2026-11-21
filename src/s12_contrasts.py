"""Stage 12: regime contrasts, seed variability, eligibility, sensitivity and sorting-error analyses. Reads the predictions and tables written by s07-s09.

  A  regime contrasts on RealWaste macro-F1 for every neural backbone:
       family A (confirmatory extension of H3): T-rand > T-paste, one-sided, Holm over backbones
       family B (exploratory decomposition):    T-shuffle - T-paste, T-rand - T-shuffle, two-sided, Holm over all
       descriptive:                             every regime minus T-orig
  B  seed-level variability (single-seed mean and SD) next to ensemble values with group-bootstrap CIs
  C  counterfactual eligibility: why each test image was excluded, per class
  D  sensitivity of BG-GAP to stricter eligibility thresholds, and a class-balanced BG-GAP
  E  sorting-relevant error rates on RealWaste (recyclable sent to trash, trash sent to recycling, per-class
     recall) and a two-stream error-cost sensitivity analysis
  F  RealWaste duplicate audit summary

Outputs: results/contrasts.json (all of the above), regime_contrasts.csv (A), seed_variability.csv (B),
exclusion_reasons.csv (C), bggap_sensitivity.csv (D), sorting_errors.csv and error_cost.csv (E).

  python -m src.s12_contrasts
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .common import CLASSES, CLS2IDX, RESULTS, WORK
from .s09_analysis import (
    B_CI,
    B_P,
    CHANCE_6,
    INFO,
    NEURAL_MODELS,
    SEEDS,
    SVM_MODELS,
    TM6,
    has_preds,
    load,
    reliance_arrays,
)
from .stats import boot_stat, ci, confusion, holm, metrics_from_cm, per_class

TRASH = CLS2IDX["trash"]
RECYCLABLE = [c for c in range(6) if c != TRASH]
REGS = ["orig", "jitter", "paste6", "shuffle6", "rand6"]
SHOW = {"orig": "T-orig", "jitter": "T-jitter", "paste6": "T-paste", "shuffle6": "T-shuffle", "rand6": "T-rand"}


def f1(y, p, w=None):
    return metrics_from_cm(confusion(y, p, w))["macro_f1"]


def rw_pred(key, reg, rep):
    return load(key, reg, rep, is_neural=True)["rw"].argmax(1)


# ------------------------------------------------------------------ A: regime contrasts ----------
def contrasts():
    y, g = INFO["rw"]
    rows = []
    pairs = [("rand6", "paste6", "A", "greater"), ("shuffle6", "paste6", "B", "two"), ("rand6", "shuffle6", "B", "two")]
    pairs += [(r, "orig", "desc", "two") for r in ("jitter", "paste6", "shuffle6", "rand6")]
    for disp, key in NEURAL_MODELS:
        for a, b, fam, side in pairs:
            if not (has_preds(key, a, True) and has_preds(key, b, True)):
                continue
            pa, pb = rw_pred(key, a, "ens"), rw_pred(key, b, "ens")
            d = boot_stat(lambda w: f1(y, pa, w) - f1(y, pb, w), g, B_P, seed=21)
            lo, hi = ci(d)
            p_le, p_ge = (1 + (d <= 0).sum()) / (B_P + 1), (1 + (d >= 0).sum()) / (B_P + 1)
            seed_d = [float(f1(y, rw_pred(key, a, s)) - f1(y, rw_pred(key, b, s))) for s in SEEDS]
            rows.append(dict(
                model=disp, contrast=f"{SHOW[a]} - {SHOW[b]}", family=fam,
                diff_ens=float(f1(y, pa) - f1(y, pb)), lo=float(lo), hi=float(hi),
                p=float(p_le if side == "greater" else min(1.0, 2 * min(p_le, p_ge))), sided=side,
                seed_diffs=seed_d, seed_mean=float(np.mean(seed_d)), seed_sd=float(np.std(seed_d, ddof=1)),
                seeds_positive=int(sum(x > 0 for x in seed_d)),
            ))
    df = pd.DataFrame(rows)
    if len(df):
        df["p_holm"] = np.nan
        for fam in ("A", "B"):
            m = df.family == fam
            if m.any():
                df.loc[m, "p_holm"] = holm(df.loc[m, "p"].tolist())
    df.to_csv(RESULTS / "regime_contrasts.csv", index=False)
    return df


# ------------------------------------------------------------------ B: seed variability ----------
def seed_table():
    y, g = INFO["rw"]
    yt, gt = INFO["tn_test"]
    rows = []
    for disp, key in NEURAL_MODELS:
        for reg in REGS:
            if not has_preds(key, reg, True):
                continue
            for name, (yy, gg), k in (("rw", (y, g), "rw"), ("tn", (yt, gt), "tn_test")):
                seeds = [float(f1(yy, load(key, reg, s, True)[k].argmax(1))) for s in SEEDS]
                pe = load(key, reg, "ens", True)[k].argmax(1)
                lo, hi = ci(boot_stat(lambda w: f1(yy, pe, w), gg, B_CI, seed=22))
                rows.append(dict(model=disp, regime=SHOW[reg], set=name, ens=float(f1(yy, pe)), ens_lo=float(lo),
                                 ens_hi=float(hi), seed_mean=float(np.mean(seeds)), seed_sd=float(np.std(seeds, ddof=1)),
                                 seed_min=min(seeds), seed_max=max(seeds)))
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "seed_variability.csv", index=False)
    return df


# ------------------------------------------------------------------ C: eligibility audit ---------
def eligibility():
    t = pd.read_csv(WORK / "poly6_table.csv")
    t = t[t.split == "test"].copy()
    crit = json.load(open(RESULTS / "variants6_info.json"))["cf_criteria"]

    def reason(r):
        if r.poly_status != "matched":
            return f"no usable polygon ({r.poly_status})"
        if r.poly_area < crit["area_min"]:
            return "object < 2% of frame"
        if r.poly_area > crit["area_max"]:
            return "object > 75% of frame"
        if r.poly_crop_loss > crit["crop_loss_max"]:
            return "> 25% of object cut by square crop"
        if r.poly_strip < crit["min_strip_px"]:
            return "background strip < 8 px"
        return "eligible"

    t["reason"] = t.apply(reason, axis=1)
    tab = pd.crosstab(t.reason, t.cls, margins=True, margins_name="total")
    tab.to_csv(RESULTS / "exclusion_reasons.csv")
    assert int((t.reason == "eligible").sum()) == int(t.cf_eligible.sum()), "eligibility reasons disagree with cf_eligible"
    return tab


# ------------------------------------------------------------------ D: BG-GAP sensitivity --------
def bg_gap_sensitivity():
    y, g = INFO["var6"]
    meta = TM6.reset_index(drop=True)
    subsets = {
        "all eligible (n=258)": np.ones(len(meta), bool),
        "object 5-60% of frame": meta.poly_area.between(0.05, 0.60).values,
        "crop loss <= 10%": (meta.poly_crop_loss <= 0.10).values,
        "background strip >= 16 px": (meta.poly_strip >= 16).values,
        "all three stricter": (meta.poly_area.between(0.05, 0.60) & (meta.poly_crop_loss <= 0.10) & (meta.poly_strip >= 16)).values,
    }
    models = [(d, k, False, "orig") for d, k in SVM_MODELS]
    models += [(d, k, True, r) for d, k in NEURAL_MODELS for r in ("orig", "paste6", "shuffle6", "rand6")]
    rows = []
    for disp, key, neural, reg in models:
        if not has_preds(key, reg, neural):
            continue
        P = load(key, reg, "ens" if neural else 0, is_neural=neural)
        if "var6_base_ORIGINAL" not in P:
            continue
        gap = reliance_arrays(P, "base", suite="v6", restrict_4way=False)["bg_gap"]
        for name, m in subsets.items():
            gm, ym, xm = g[m], y[m], gap[m]
            d = boot_stat(lambda w: (w @ xm) / np.maximum(w.sum(1), 1e-9), gm, B_CI, seed=23)
            lo, hi = ci(d)
            rows.append(dict(model=disp, regime=SHOW[reg], subset=name, n=int(m.sum()),
                             n_by_class={CLASSES[c]: int((ym == c).sum()) for c in range(6)},
                             bg_gap=float(xm.mean()), lo=float(lo), hi=float(hi)))
        # class-balanced BG-GAP: mean of the six per-class means
        cls_masks = [y == c for c in range(6)]

        def cb(w):
            return np.mean([(w[:, m] @ gap[m]) / np.maximum(w[:, m].sum(1), 1e-9) for m in cls_masks], 0)

        lo, hi = ci(boot_stat(cb, g, B_CI, seed=24))
        rows.append(dict(model=disp, regime=SHOW[reg], subset="class-balanced (mean of 6 class means)", n=len(y),
                         n_by_class=None, bg_gap=float(np.mean([gap[m].mean() for m in cls_masks])), lo=float(lo), hi=float(hi)))
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "bggap_sensitivity.csv", index=False)
    return df


# ------------------------------------------------------------------ E: sorting errors + cost -----
def sorting_errors():
    y, g = INFO["rw"]
    rec = np.isin(y, RECYCLABLE)
    rows, cost_rows = [], []
    ratios = [0.5, 1, 2, 4]
    for disp, key in NEURAL_MODELS:
        for reg in REGS:
            if not has_preds(key, reg, True):
                continue
            p = rw_pred(key, reg, "ens")
            to_trash = rec & (p == TRASH)          # recyclable lost to landfill
            to_recycling = (~rec) & (p != TRASH)   # trash contaminating recycling
            wrong_bin = rec & (p != TRASH) & (p != y)  # recyclable, right stream, wrong material

            def rates(w):
                return np.stack([(w[:, rec] @ to_trash[rec]) / w[:, rec].sum(1),
                                 (w[:, ~rec] @ to_recycling[~rec]) / w[:, ~rec].sum(1),
                                 (w[:, rec] @ wrong_bin[rec]) / w[:, rec].sum(1)], 1)

            lo, hi = ci(boot_stat(rates, g, B_CI, seed=25))
            _, recall, _ = per_class(confusion(y, p))
            rows.append(dict(model=disp, regime=SHOW[reg], n_recyclable=int(rec.sum()), n_trash=int((~rec).sum()),
                             recyclable_to_trash=int(to_trash.sum()), recyclable_to_trash_rate=float(to_trash[rec].mean()),
                             r2t_lo=float(lo[0]), r2t_hi=float(hi[0]),
                             trash_to_recycling=int(to_recycling.sum()), trash_to_recycling_rate=float(to_recycling[~rec].mean()),
                             t2r_lo=float(lo[1]), t2r_hi=float(hi[1]),
                             wrong_material=int(wrong_bin.sum()), wrong_material_rate=float(wrong_bin[rec].mean()),
                             wm_lo=float(lo[2]), wm_hi=float(hi[2]),
                             **{f"recall_{CLASSES[c]}": float(recall[c]) for c in range(6)}))
            # Two-stream cost per 1,000 images: a lost recyclable costs r, a contaminating trash item costs 1.
            for r in ratios:
                cost_rows.append(dict(model=disp, regime=SHOW[reg], ratio_lost_to_contam=r,
                                      cost_per_1000=float(1000 * (r * to_trash.sum() + to_recycling.sum()) / len(y))))
    df, cost = pd.DataFrame(rows), pd.DataFrame(cost_rows)
    df.to_csv(RESULTS / "sorting_errors.csv", index=False)
    cost.to_csv(RESULTS / "error_cost.csv", index=False)
    return df, cost


# ------------------------------------------------------------------ F: RealWaste duplicates ------
def realwaste_duplicates():
    rw = pd.read_csv(WORK / "realwaste_groups.csv")
    sizes = rw.group.value_counts()
    multi = sizes[sizes > 1]
    pairs = pd.read_csv(WORK / "realwaste_hash_pairs.csv") if (WORK / "realwaste_hash_pairs.csv").exists() else None
    return dict(n_images=int(len(rw)), n_groups=int(rw.group.nunique()), multi_member_groups=int(len(multi)),
                images_in_multi_groups=int(multi.sum()), largest_group=int(sizes.max()),
                label_conflicts=int(rw.drop_label_conflict.sum()),
                hash_pairs=None if pairs is None else int(len(pairs)))


def main():
    out = {}
    c = contrasts()
    out["contrasts"] = c.to_dict(orient="records")
    s = seed_table()
    out["seed_variability"] = s.to_dict(orient="records")
    out["eligibility"] = eligibility().to_dict()
    out["bg_gap_sensitivity"] = bg_gap_sensitivity().to_dict(orient="records")
    e, cost = sorting_errors()
    out["sorting_errors"] = e.to_dict(orient="records")
    out["error_cost"] = cost.to_dict(orient="records")
    out["realwaste_duplicates"] = realwaste_duplicates()
    json.dump(out, open(RESULTS / "contrasts.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 60)
    print(c.drop(columns=["seed_diffs"]).round(4).to_string())
    print(out["realwaste_duplicates"])


if __name__ == "__main__":
    main()
