"""Stage 9: metrics, background reliance, probes and the planned tests.

For every model (majority class, colour and SIFT SVMs, four networks), training regime and seed, plus the
3-seed softmax ensemble of each network:
  - accuracy, macro-F1, balanced accuracy and per-class scores with duplicate-group bootstrap 95% CIs on the
    TrashNet test split, the edited-set originals and RealWaste (six-class and five-class without trash);
  - background reliance on edited test images (BG-GAP, flip rate, chance-corrected BG-GAP with chance 1/6):
    six-class polygon suite (paper) and four-class U2-Net suite (supplementary);
  - background-only probes with duplicate-group label-permutation tests;
  - planned tests H1-H3 for the partly frozen ResNet-18 with Holm correction, secondary tests, RealWaste
    error types.

Reads results/preds/*.npz, data/work/ tables and results/rq4.json. Writes results/performance.csv,
performance_seed_stats.csv, confusion_matrices.npy, reliance6.csv, reliance6_per_class.csv, reliance.csv,
reliance_per_class.csv, probes.csv, mask_quality_covariate.csv, error_modes.csv and summary.json.

  python -m src.s09_analysis
"""
from __future__ import annotations

import json
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .common import CLASSES, CLS2IDX, N_PAIRINGS, RESULTS, WORK
from .s05_variants import VDIR
from .s05_variants6 import V6DIR
from .stats import (
    boot_stat,
    ci,
    confusion,
    holm,
    mcnemar_exact,
    metrics_from_cm,
    per_class,
    perm_test_balanced_acc,
    sign_flip_test,
    wmean,
)
from .testsets import MASK_KINDS

PRED = RESULTS / "preds"
B_CI, B_P = 2000, 10_000
SEEDS = (0, 1, 2)
TRASH = CLS2IDX["trash"]

NEURAL_MODELS = [
    ("M2 ResNet-18", "cnn"),
    ("M2b ResNet-18 (full-FT)", "resnet18_ft"),
    ("M3 MobileNetV3-L", "mobilenet_v3"),
    ("M4 Swin-T", "swin_t"),
]
SVM_MODELS = [
    ("M1a colour SVM", "svm_hist"),
    ("M1b SIFT SVM", "svm_sift"),
]
# Regime labels written to every table; they match the paper. The *-U2Net regimes are the supplementary
# four-class suite. The order of this dict sets the row order of the tables.
ALL_REGIMES = {
    "orig": "T-orig",
    "jitter": "T-jitter",
    "paste": "T-paste-U2Net (suppl.)",
    "rand": "T-rand-U2Net (suppl.)",
    "paste6": "T-paste",
    "rand6": "T-rand",
    "shuffle6": "T-shuffle",
}


def set_info():
    sp = pd.read_csv(WORK / "split.csv")
    rw = pd.read_csv(WORK / "realwaste_groups.csv")
    tm = pd.read_csv(VDIR / "test_eligible_meta.csv")
    keep = np.load(VDIR / "test_fixedbox_keep.npy")
    lab = lambda m: m.cls.map(CLS2IDX).values
    info = {}
    for s in ("val", "test"):
        m = sp[sp.split == s]
        info[f"tn_{s}"] = (lab(m), m.group.values)
    info["rw"] = (lab(rw), rw.group.values)
    info["var"] = (lab(tm), tm.group.values)
    info["probe_fixed"] = (lab(tm)[keep], tm.group.values[keep])

    tm6 = None
    if (V6DIR / "test_cf_meta.csv").exists():
        tm6 = pd.read_csv(V6DIR / "test_cf_meta.csv")
        cm6 = pd.read_csv(V6DIR / "test_corner_meta.csv")
        info["var6"] = (lab(tm6), tm6.group.values)
        info["probe_corners"] = (lab(cm6), cm6.group.values)
    return info, tm, tm6


INFO, TM, TM6 = set_info()
VAR4_CLASSES = sorted(set(INFO["var"][0].tolist()))
CHANCE_6 = 1.0 / 6.0
CHANCE_4 = 1.0 / len(VAR4_CLASSES)


def has_preds(key: str, regime: str, is_neural: bool) -> bool:
    if key == "m0":
        return (PRED / "m0_orig_d0.npz").exists() and regime == "orig"
    if is_neural:
        return all((PRED / f"{key}_{regime}_s{s}.npz").exists() for s in SEEDS)
    if regime in ("orig", "probe_bg", "probe_fixed", "probe_bg6", "probe_corners"):
        return (PRED / f"{key}_{regime}_d0.npz").exists()
    return all((PRED / f"{key}_{regime}_d{d}.npz").exists() for d in SEEDS)


def load(key: str, regime: str, rep, is_neural: bool | None = None) -> dict[str, np.ndarray]:
    if is_neural is None:
        is_neural = key in {k for _, k in NEURAL_MODELS}
    if is_neural:
        if rep == "ens":
            zs = [np.load(PRED / f"{key}_{regime}_s{s}.npz") for s in SEEDS]
            common_keys = set(zs[0].files).intersection(*(z.files for z in zs[1:]))
            return {k: np.mean([z[k] for z in zs], 0) for k in sorted(common_keys)}
        z = np.load(PRED / f"{key}_{regime}_s{rep}.npz")
        return {k: z[k] for k in z.files}
    z = np.load(PRED / f"{key}_{regime}_d{rep}.npz")
    return {k: z[k] for k in z.files}


def labels_of(arr: np.ndarray, allowed_classes: list[int] | None = None) -> np.ndarray:
    if arr.ndim == 2:
        if allowed_classes is not None:
            sub = arr[:, allowed_classes]
            return np.asarray(allowed_classes)[sub.argmax(1)]
        return arr.argmax(1)
    return arr


# ---------------------------------------------------------------- performance -------------------
def perf_rows(name: str, y: np.ndarray, p: np.ndarray, groups: np.ndarray, **tags) -> dict:
    cm = confusion(y, p)
    point = metrics_from_cm(cm)
    draws = boot_stat(lambda w: np.stack(list(metrics_from_cm(confusion(y, p, w)).values()), 1), groups, B_CI)
    lo, hi = ci(draws)
    row = dict(tags, set=name, n=len(y))
    for k, (m, v) in enumerate(point.items()):
        row[m] = v
        row[f"{m}_lo"] = lo[k]
        row[f"{m}_hi"] = hi[k]
    prec, rec, f1 = per_class(cm)
    for c, cl in enumerate(CLASSES):
        row[f"recall_{cl}"] = rec[c]
        row[f"f1_{cl}"] = f1[c]
    return row


def performance():
    rows, cms = [], {}
    all_models = [("M0 majority", "m0", False)] + [(d, k, False) for d, k in SVM_MODELS] + [(d, k, True) for d, k in NEURAL_MODELS]
    for disp, key, is_neural in all_models:
        for reg, regname in ALL_REGIMES.items():
            if not has_preds(key, reg, is_neural):
                continue
            reps = ["ens", 0, 1, 2] if is_neural else ([0] if reg == "orig" else [0, 1, 2])
            for rep in reps:
                P = load(key, reg, rep, is_neural=is_neural)
                tags = dict(model=disp, regime=regname, rep=str(rep))
                eval_targets = [("tn_test", "tn_test"), ("var_base_ORIGINAL", "tn_test_eligible"), ("rw", "rw")]
                if "var6_base_ORIGINAL" in P:
                    eval_targets.append(("var6_base_ORIGINAL", "tn_test_cf6"))
                for pkey, out_name in eval_targets:
                    if pkey not in P:
                        continue
                    info_key = "var" if pkey.startswith("var_") else ("var6" if pkey.startswith("var6_") else pkey)
                    y, g = INFO[info_key]
                    pr = labels_of(P[pkey])
                    rows.append(perf_rows(out_name, y, pr, g, **tags))
                    if out_name in ("tn_test", "rw") and str(rep) in ("ens", "0"):
                        cms[(disp, regname, str(rep), out_name)] = confusion(y, pr)

                # Five-class sensitivity on RealWaste (drop trash, argmax over 5 material classes)
                y, g = INFO["rw"]
                keep = y != TRASH
                if is_neural:
                    p5 = P["rw"][keep][:, :5].argmax(1)
                elif key == "m0":
                    p5 = P["rw"][keep]
                else:
                    p5 = P["rw_dec"][keep][:, :5].argmax(1)
                rows.append(perf_rows("rw_5class", y[keep], p5, g[keep], **tags))

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "performance.csv", index=False)
    np.save(RESULTS / "confusion_matrices.npy", {"|".join(k): v for k, v in cms.items()}, allow_pickle=True)

    # Seed mean ± SD summary alongside ensemble
    seed_rows = []
    for (m, reg, sname), gdf in df[df.rep.isin(["0", "1", "2"])].groupby(["model", "regime", "set"]):
        if len(gdf) < 2:
            continue
        ens_df = df[(df.model == m) & (df.regime == reg) & (df.set == sname) & (df.rep == "ens")]
        seed_rows.append(
            dict(
                model=m,
                regime=reg,
                set=sname,
                n_seeds=len(gdf),
                seed_macro_f1_mean=float(gdf.macro_f1.mean()),
                seed_macro_f1_sd=float(gdf.macro_f1.std(ddof=1)),
                seed_acc_mean=float(gdf.accuracy.mean()),
                seed_acc_sd=float(gdf.accuracy.std(ddof=1)),
                ens_macro_f1=float(ens_df.macro_f1.iloc[0]) if len(ens_df) else np.nan,
                ens_macro_f1_lo=float(ens_df.macro_f1_lo.iloc[0]) if len(ens_df) else np.nan,
                ens_macro_f1_hi=float(ens_df.macro_f1_hi.iloc[0]) if len(ens_df) else np.nan,
                ens_acc=float(ens_df.accuracy.iloc[0]) if len(ens_df) else np.nan,
            )
        )
    pd.DataFrame(seed_rows).to_csv(RESULTS / "performance_seed_stats.csv", index=False)
    return df, cms


# ---------------------------------------------------------------- reliance (H2) -----------------
def reliance_arrays(P: dict, kind: str, suite: str = "v4", restrict_4way: bool = False) -> dict:
    prefix = "var" if suite == "v4" else "var6"
    vdir = VDIR if suite == "v4" else V6DIR
    tm_df = TM if suite == "v4" else TM6
    y = INFO[prefix][0]
    pairs = json.load(open(vdir / "test_pairings.json"))
    cls_of = dict(zip(tm_df.id, tm_df.cls.map(CLS2IDX)))
    allowed = VAR4_CLASSES if (suite == "v4" and restrict_4way) else None

    pr = lambda v: labels_of(P[f"{prefix}_{'base' if v == 'ORIGINAL' else kind}_{v}"], allowed_classes=allowed)
    o, fg, bg = pr("ORIGINAL"), pr("ONLY_FG"), pr("ONLY_BG_T")
    same = np.stack([pr(f"MIXED_SAME_{p}") for p in range(N_PAIRINGS)])
    rand = np.stack([pr(f"MIXED_RAND_{p}") for p in range(N_PAIRINGS)])
    donor = np.stack([[cls_of[i] for i in pairs[str(p)]["rand"]] for p in range(N_PAIRINGS)])
    a = {
        "acc_ORIGINAL": (o == y),
        "acc_ONLY_FG": (fg == y),
        "acc_ONLY_BG_T": (bg == y),
        "acc_MIXED_SAME": (same == y).mean(0),
        "acc_MIXED_RAND": (rand == y).mean(0),
        "flip_rate": (same != rand).mean(0),
        "change_ONLY_FG": (fg != o),
        "change_ONLY_BG_T": (bg != o),
        "change_MIXED_SAME": (same != o).mean(0),
        "change_MIXED_RAND": (rand != o).mean(0),
        "pull_RAND": (rand == donor).mean(0),
        "pull_SAME_ref": (same == donor).mean(0),
    }
    a = {k: v.astype(np.float64) for k, v in a.items()}
    a["bg_gap"] = a["acc_MIXED_SAME"] - a["acc_MIXED_RAND"]
    return a


def summarise_reliance(a: dict, groups: np.ndarray, chance: float, B: int = B_CI, seed: int = 0):
    keys = list(a)
    X = np.stack([a[k] for k in keys], 1)
    draws = boot_stat(lambda w: wmean(X, w), groups, B, seed)
    lo, hi = ci(draws)
    out = {}
    for j, k in enumerate(keys):
        out[k] = float(X[:, j].mean())
        out[f"{k}_lo"] = float(lo[j])
        out[f"{k}_hi"] = float(hi[j])
    js, jg = keys.index("acc_MIXED_SAME"), keys.index("bg_gap")
    denom = np.maximum(draws[:, js] - chance, 1e-6)
    cc = draws[:, jg] / denom
    out["chance_baseline"] = chance
    out["cc_bg_gap"] = float(X[:, jg].mean() / max(X[:, js].mean() - chance, 1e-6))
    out["cc_bg_gap_lo"], out["cc_bg_gap_hi"] = map(float, ci(cc))
    return out, cc


REL_KEYS = ("flip_rate", "bg_gap", "acc_ORIGINAL", "acc_MIXED_SAME", "acc_MIXED_RAND")


def run_reliance_suite(suite: str = "v4"):
    prefix = "var" if suite == "v4" else "var6"
    if prefix not in INFO:
        return pd.DataFrame(), {}, {}
    y, groups = INFO[prefix]
    present_classes = sorted(set(y.tolist()))
    rows, cc_draws, pvals, per_cls = [], {}, {}, []
    models = [(d, k, False) for d, k in SVM_MODELS] + [(d, k, True) for d, k in NEURAL_MODELS]

    for disp, key, is_neural in models:
        for reg, regname in ALL_REGIMES.items():
            if not has_preds(key, reg, is_neural):
                continue
            reps = ["ens", 0, 1, 2] if is_neural else ([0] if reg == "orig" else [0, 1, 2])
            for rep in reps:
                P = load(key, reg, rep, is_neural=is_neural)
                if f"{prefix}_base_ORIGINAL" not in P:
                    continue
                primary_rep = str(rep) == ("ens" if is_neural else "0")
                for kind in MASK_KINDS:
                    if kind != "base" and not (reg == "orig" and primary_rep):
                        continue
                    # Six-way predictions, so chance is 1/6
                    a = reliance_arrays(P, kind, suite=suite, restrict_4way=False)
                    s, cc = summarise_reliance(a, groups, chance=CHANCE_6, seed=0)
                    row = dict(suite=suite, model=disp, regime=regname, rep=str(rep), masks=kind, n=len(y), **s)
                    # Four-class suite, networks: also the BG-GAP with the softmax restricted to the four classes
                    if suite == "v4" and is_neural and kind == "base":
                        a4 = reliance_arrays(P, kind, suite="v4", restrict_4way=True)
                        s4, _ = summarise_reliance(a4, groups, chance=CHANCE_4, seed=0)
                        row["bg_gap_4way_restricted"] = s4["bg_gap"]
                        row["cc_bg_gap_4way_restricted"] = s4["cc_bg_gap"]
                    if reg == "orig" and primary_rep:
                        _, p = sign_flip_test(a["bg_gap"], groups, B_P, seed=1)
                        row["bg_gap_signflip_p"] = float(p)
                        if kind == "base":
                            cc_draws[disp] = (s["cc_bg_gap"], cc)
                            pvals[disp] = float(p)
                    rows.append(row)
                    if kind == "base" and primary_rep:
                        for c in present_classes:
                            cl, m = CLASSES[c], y == c
                            sub = {k: v[m] for k, v in a.items()}
                            sc, _ = summarise_reliance(sub, groups[m], chance=CHANCE_6, B=1000)
                            per_cls.append(
                                dict(
                                    suite=suite,
                                    model=disp,
                                    regime=regname,
                                    rep=str(rep),
                                    cls=cl,
                                    n=int(m.sum()),
                                    **{f"{k}{s_}": sc[f"{k}{s_}"] for k in REL_KEYS for s_ in ("", "_lo", "_hi")},
                                )
                            )
    out_df = pd.DataFrame(rows)
    cls_df = pd.DataFrame(per_cls)
    if suite == "v4":
        out_df.to_csv(RESULTS / "reliance.csv", index=False)
        cls_df.to_csv(RESULTS / "reliance_per_class.csv", index=False)
    else:
        out_df.to_csv(RESULTS / "reliance6.csv", index=False)
        cls_df.to_csv(RESULTS / "reliance6_per_class.csv", index=False)
    return out_df, cc_draws, pvals


def mask_quality_covariate():
    mt = pd.read_csv(WORK / "mask_table.csv").set_index("id")
    iou = mt.loc[TM.id, "u2net_iou"].values
    ok = ~np.isnan(iou)
    out = []
    for disp, key, rep, is_n in [
        ("M2 ResNet-18", "cnn", "ens", True),
        ("M1a colour SVM", "svm_hist", 0, False),
        ("M1b SIFT SVM", "svm_sift", 0, False),
    ]:
        if not has_preds(key, "orig", is_n):
            continue
        a = reliance_arrays(load(key, "orig", rep, is_neural=is_n), "base", suite="v4")
        rho, p = spearmanr(iou[ok], a["flip_rate"][ok])
        terc = pd.qcut(iou[ok], 3, labels=["low", "mid", "high"])
        fr = pd.Series(a["flip_rate"][ok]).groupby(terc, observed=True).mean()
        verified = ok & (np.nan_to_num(iou) >= 0.85)
        s, _ = summarise_reliance({k: v[verified] for k, v in a.items()}, INFO["var"][1][verified], chance=CHANCE_6, B=1000)
        out.append(
            dict(
                model=disp,
                n_with_reference=int(ok.sum()),
                spearman_rho_iou_vs_flip=float(rho),
                spearman_p=float(p),
                flip_low_iou=float(fr.get("low")),
                flip_mid_iou=float(fr.get("mid")),
                flip_high_iou=float(fr.get("high")),
                verified_n=int(verified.sum()),
                verified_flip=s["flip_rate"],
                verified_flip_lo=s["flip_rate_lo"],
                verified_flip_hi=s["flip_rate_hi"],
                verified_bg_gap=s["bg_gap"],
                verified_bg_gap_lo=s["bg_gap_lo"],
                verified_bg_gap_hi=s["bg_gap_hi"],
            )
        )
    res = pd.DataFrame(out)
    res.to_csv(RESULTS / "mask_quality_covariate.csv", index=False)
    return res


# ---------------------------------------------------------------- probes (H1) ---------------------
def probes():
    rows, pv = [], {}
    probe_specs = [
        ("bg", "var", CHANCE_4, [("base", "var_base_ONLY_BG_T", "probe"), ("erode3", "var_erode3_ONLY_BG_T", "probe_erode3"), ("dilate3", "var_dilate3_ONLY_BG_T", "probe_dilate3")]),
        ("fixed", "probe_fixed", CHANCE_4, [("base", "probe_fixed", "probe")]),
        ("bg6", "var6", CHANCE_6, [("base", "var6_base_ONLY_BG_T", "probe"), ("erode3", "var6_erode3_ONLY_BG_T", "probe_erode3"), ("dilate3", "var6_dilate3_ONLY_BG_T", "probe_dilate3")]),
        ("corners", "probe_corners", CHANCE_6, [("base", "probe_corners", "probe")]),
    ]
    models = [(d, k, True) for d, k in NEURAL_MODELS] + [(d, k, False) for d, k in SVM_MODELS]
    for disp, key, is_neural in models:
        reps = ["ens", 0, 1, 2] if is_neural else [0]
        for kind, info_key, ch_val, test_list in probe_specs:
            if info_key not in INFO or not has_preds(key, f"probe_{kind}", is_neural):
                continue
            y, g = INFO[info_key]
            for rep in reps:
                P = load(key, f"probe_{kind}", rep, is_neural=is_neural)
                for masks, n_key, s_key in test_list:
                    arr_key = n_key if is_neural else s_key
                    if arr_key not in P:
                        continue
                    p = labels_of(P[arr_key])
                    obs, perm_p, _ = perm_test_balanced_acc(
                        y, p, n=B_P if (masks == "base") else 2000, seed=2, groups=g
                    )
                    d = boot_stat(lambda w: metrics_from_cm(confusion(y, p, w))["balanced_acc"], g, B_CI)
                    lo, hi = ci(d)
                    rows.append(
                        dict(
                            model=disp,
                            rep=str(rep),
                            probe=kind,
                            masks=masks,
                            n=len(y),
                            balanced_acc=float(obs),
                            ci_lo=float(lo),
                            ci_hi=float(hi),
                            perm_p=float(perm_p),
                            chance=float(ch_val),
                            majority_rate=float(np.bincount(y).max() / len(y)),
                            accuracy=float((p == y).mean()),
                        )
                    )
                    if masks == "base":
                        pv[(disp, str(rep), kind)] = float(perm_p)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "probes.csv", index=False)
    return df, pv


# ---------------------------------------------------------------- RealWaste transfer (H3) ---------
def realwaste_regime_comparison(key: str, rand_reg: str, paste_reg: str, five: bool = False) -> dict | None:
    if not (has_preds(key, rand_reg, True) and has_preds(key, paste_reg, True) and has_preds(key, "orig", True)):
        return None
    y, g = INFO["rw"]
    keep = (y != TRASH) if five else np.ones(len(y), bool)
    y, g = y[keep], g[keep]
    k_used = 5 if five else 6

    def preds(reg, rep):
        pr = load(key, reg, rep, is_neural=True)["rw"][keep]
        return pr[:, :k_used].argmax(1)

    f = lambda p, w=None: metrics_from_cm(confusion(y, p, w))["macro_f1"]
    pa, ra, oa = preds(paste_reg, "ens"), preds(rand_reg, "ens"), preds("orig", "ens")
    diff_draws = boot_stat(lambda w: f(ra, w) - f(pa, w), g, B_P, seed=4)
    orig_draws = boot_stat(lambda w: f(ra, w) - f(oa, w), g, B_CI, seed=8)
    paste_vs_orig = boot_stat(lambda w: f(pa, w) - f(oa, w), g, B_CI, seed=9)

    out = {
        "diff_rand_minus_paste_ens": float(f(ra) - f(pa)),
        "diff_lo": float(ci(diff_draws)[0]),
        "diff_hi": float(ci(diff_draws)[1]),
        "p_one_sided": float((1 + (diff_draws <= 0).sum()) / (B_P + 1)),
        "per_seed_diff": [float(f(preds(rand_reg, s)) - f(preds(paste_reg, s))) for s in SEEDS],
        "diff_rand_minus_orig_ens": float(f(ra) - f(oa)),
        "diff_rand_minus_orig_lo": float(ci(orig_draws)[0]),
        "diff_rand_minus_orig_hi": float(ci(orig_draws)[1]),
        "per_seed_rand_minus_orig": [float(f(preds(rand_reg, s)) - f(preds("orig", s))) for s in SEEDS],
        "diff_paste_minus_orig_ens": float(f(pa) - f(oa)),
        "diff_paste_minus_orig_lo": float(ci(paste_vs_orig)[0]),
        "diff_paste_minus_orig_hi": float(ci(paste_vs_orig)[1]),
        "macro_f1_orig_ens": float(f(oa)),
        "macro_f1_orig_seeds": [float(f(preds("orig", s))) for s in SEEDS],
        "macro_f1_paste_ens": float(f(pa)),
        "macro_f1_paste_seeds": [float(f(preds(paste_reg, s))) for s in SEEDS],
        "macro_f1_rand_ens": float(f(ra)),
        "macro_f1_rand_seeds": [float(f(preds(rand_reg, s))) for s in SEEDS],
        "mcnemar_b_c_p": mcnemar_exact(ra == y, pa == y),
    }
    out["seeds_agree_positive"] = all(d > 0 for d in out["per_seed_diff"])
    if has_preds(key, "jitter", True):
        ja = preds("jitter", "ens")
        jit_draws = boot_stat(lambda w: f(ja, w) - f(oa, w), g, B_CI, seed=10)
        out["macro_f1_jitter_ens"] = float(f(ja))
        out["macro_f1_jitter_seeds"] = [float(f(preds("jitter", s))) for s in SEEDS]
        out["diff_jitter_minus_orig_ens"] = float(f(ja) - f(oa))
        out["diff_jitter_minus_orig_lo"] = float(ci(jit_draws)[0])
        out["diff_jitter_minus_orig_hi"] = float(ci(jit_draws)[1])
    return out


def gap_closed_calc(key: str = "cnn", rand_reg: str = "rand", paste_reg: str = "paste") -> dict | None:
    if not (has_preds(key, rand_reg, True) and has_preds(key, paste_reg, True) and has_preds(key, "orig", True)):
        return None
    yt, gt = INFO["tn_test"]
    yr, gr = INFO["rw"]
    P = {r: load(key, r, "ens", is_neural=True) for r in ("orig", paste_reg, rand_reg)}
    f = lambda y, p, w=None: metrics_from_cm(confusion(y, p, w))["macro_f1"]
    pt = {r: P[r]["tn_test"].argmax(1) for r in P}
    prw = {r: P[r]["rw"].argmax(1) for r in P}
    tn_d = boot_stat(lambda w: np.stack([f(yt, pt["orig"], w), f(yt, pt[paste_reg], w), f(yt, pt[rand_reg], w)], 1), gt, B_CI, seed=5)
    rw_d = boot_stat(lambda w: np.stack([f(yr, prw["orig"], w), f(yr, prw[paste_reg], w), f(yr, prw[rand_reg], w)], 1), gr, B_CI, seed=6)
    share_vs_paste = (rw_d[:, 2] - rw_d[:, 1]) / (tn_d[:, 1] - rw_d[:, 1])
    share_vs_orig = (rw_d[:, 2] - rw_d[:, 0]) / (tn_d[:, 0] - rw_d[:, 0])
    return {
        "share_closed_vs_paste": float((f(yr, prw[rand_reg]) - f(yr, prw[paste_reg])) / (f(yt, pt[paste_reg]) - f(yr, prw[paste_reg]))),
        "vs_paste_lo": float(ci(share_vs_paste)[0]),
        "vs_paste_hi": float(ci(share_vs_paste)[1]),
        "share_closed_vs_orig": float((f(yr, prw[rand_reg]) - f(yr, prw["orig"])) / (f(yt, pt["orig"]) - f(yr, prw["orig"])),),
        "vs_orig_lo": float(ci(share_vs_orig)[0]),
        "vs_orig_hi": float(ci(share_vs_orig)[1]),
        "tn_orig_f1": float(f(yt, pt["orig"])),
        "rw_orig_f1": float(f(yr, prw["orig"])),
    }


# ---------------------------------------------------------------- error types ---------------------
def error_mode_decomposition(cms: dict) -> pd.DataFrame:
    """Split the off-diagonal errors of each confusion matrix into five error types."""
    # Indices: 0=cardboard, 1=glass, 2=metal, 3=paper, 4=plastic, 5=trash
    SPECULAR = {1, 2, 4}
    FIBROUS = {0, 3}
    rows = []
    for (model, regime, rep, ds), cm in cms.items():
        cm = np.asarray(cm, dtype=int)
        total = int(cm.sum())
        correct = int(np.diag(cm).sum())
        errors = total - correct
        mat_to_trash = 0
        trash_to_mat = 0
        specular_conf = 0
        fibrous_conf = 0
        cross_family = 0
        for i in range(6):
            for j in range(6):
                if i == j or cm[i, j] == 0:
                    continue
                cnt = int(cm[i, j])
                if i != 5 and j == 5:
                    mat_to_trash += cnt
                elif i == 5 and j != 5:
                    trash_to_mat += cnt
                elif i in SPECULAR and j in SPECULAR:
                    specular_conf += cnt
                elif i in FIBROUS and j in FIBROUS:
                    fibrous_conf += cnt
                else:
                    cross_family += cnt
        assert mat_to_trash + trash_to_mat + specular_conf + fibrous_conf + cross_family == errors
        rows.append(
            dict(
                model=model,
                regime=regime,
                rep=rep,
                set=ds,
                n_total=total,
                n_errors=errors,
                error_rate=errors / total,
                recyclable_called_trash=mat_to_trash,
                share_recyclable_called_trash=mat_to_trash / max(errors, 1),
                trash_called_recyclable=trash_to_mat,
                share_trash_called_recyclable=trash_to_mat / max(errors, 1),
                specular_rigid_confusion=specular_conf,
                share_specular_rigid=specular_conf / max(errors, 1),
                fibrous_paper_cardboard=fibrous_conf,
                share_fibrous=fibrous_conf / max(errors, 1),
                other_cross_material=cross_family,
                share_other=cross_family / max(errors, 1),
            )
        )
    edf = pd.DataFrame(rows)
    edf.to_csv(RESULTS / "error_modes.csv", index=False)
    return edf


def main():
    perf, cms = performance()
    rel4, cc4, p2_4 = run_reliance_suite("v4")
    rel6, cc6, p2_6 = run_reliance_suite("v6")
    pr, p1 = probes()
    mq = mask_quality_covariate()
    err_df = error_mode_decomposition(cms)

    summary: dict = {}

    # Planned tests H1-H3: partly frozen ResNet-18, six-class polygon suite (paper)
    h3 = realwaste_regime_comparison("cnn", "rand6", "paste6", five=False)
    h3_5 = realwaste_regime_comparison("cnn", "rand6", "paste6", five=True)
    planned = {
        "H1": max(p1[("M2 ResNet-18", "ens", "bg6")], p1[("M2 ResNet-18", "ens", "corners")]),
        "H2": p2_6["M2 ResNet-18"],
        "H3": h3["p_one_sided"],
    }
    adj = dict(zip(planned, holm(list(planned.values()))))
    summary["planned_tests"] = {k: dict(p=planned[k], p_holm=float(adj[k])) for k in planned}
    summary["planned_tests"]["H3"]["seeds_agree_positive"] = h3["seeds_agree_positive"]
    summary["H3_six_class"] = h3
    summary["H3_five_class"] = h3_5
    summary["gap_closed"] = gap_closed_calc("cnn", "rand6", "paste6")

    # RealWaste regime comparison for every network (six-class regimes)
    multi_arch = {}
    for disp, key in NEURAL_MODELS:
        comp6 = realwaste_regime_comparison(key, "rand6", "paste6", five=False)
        if comp6 is not None:
            multi_arch[disp] = {
                "realwaste_6class": comp6,
                "realwaste_5class": realwaste_regime_comparison(key, "rand6", "paste6", five=True),
                "gap_closed_6class": gap_closed_calc(key, "rand6", "paste6"),
            }
    summary["multi_arch"] = multi_arch

    # Secondary family (six-class suite)
    sec = {}
    order = ["M1a colour SVM", "M2 ResNet-18", "M1b SIFT SVM"]
    for a, b in combinations(order, 2):
        if a in cc6 and b in cc6:
            d = cc6[a][1] - cc6[b][1]
            sec[f"RQ2 cc-BG-GAP {a.split()[0]} > {b.split()[0]}"] = dict(
                diff=float(cc6[a][0] - cc6[b][0]),
                lo=float(ci(d)[0]),
                hi=float(ci(d)[1]),
                p=float((1 + (d <= 0).sum()) / (len(d) + 1)),
            )
    for m in ("M1a colour SVM", "M1b SIFT SVM"):
        if (m, "0", "bg6") in p1 and (m, "0", "corners") in p1:
            sec[f"H1 analogue {m.split()[0]}"] = dict(p=max(p1[(m, "0", "bg6")], p1[(m, "0", "corners")]))
        if m in p2_6:
            sec[f"H2 analogue {m.split()[0]}"] = dict(p=p2_6[m])

    yt = INFO["tn_test"][0]
    preds_tn = {
        "M2 ResNet-18": load("cnn", "orig", "ens", is_neural=True)["tn_test"].argmax(1),
        "M1a colour SVM": load("svm_hist", "orig", 0, is_neural=False)["tn_test"],
        "M1b SIFT SVM": load("svm_sift", "orig", 0, is_neural=False)["tn_test"],
    }
    for a, b in [("M2 ResNet-18", "M1a colour SVM"), ("M2 ResNet-18", "M1b SIFT SVM"), ("M1a colour SVM", "M1b SIFT SVM")]:
        bb, cc_cnt, p = mcnemar_exact(preds_tn[a] == yt, preds_tn[b] == yt)
        sec[f"McNemar TrashNet test {a.split()[0]} vs {b.split()[0]}"] = dict(b=bb, c=cc_cnt, p=float(p))

    bb, cc_cnt, p = h3["mcnemar_b_c_p"]
    sec["McNemar RealWaste T-rand vs T-paste (M2 ens)"] = dict(b=bb, c=cc_cnt, p=float(p))

    rq4 = RESULTS / "rq4.json"
    if rq4.exists():
        for k, v in json.load(open(rq4)).items():
            sec[f"RQ4 {k}"] = v
    sadj = holm([v["p"] for v in sec.values()])
    for (k, v), a in zip(sec.items(), sadj):
        v["p_holm"] = float(a)

    summary["secondary"] = sec

    # Supplementary: the same planned tests on the four-class U2-Net suite (not reported in the paper)
    h3_v4 = realwaste_regime_comparison("cnn", "rand", "paste", five=False)
    if h3_v4 is not None:
        sup = {
            "H1": max(p1[("M2 ResNet-18", "ens", "bg")], p1[("M2 ResNet-18", "ens", "fixed")]),
            "H2": p2_4["M2 ResNet-18"],
            "H3": h3_v4["p_one_sided"],
        }
        adj4 = dict(zip(sup, holm(list(sup.values()))))
        tests = {k: dict(p=sup[k], p_holm=float(adj4[k])) for k in sup}
        tests["H3"]["seeds_agree_positive"] = h3_v4["seeds_agree_positive"]
        summary["supplementary_u2net"] = {
            "planned_tests": tests,
            "H3_six_class": h3_v4,
            "H3_five_class": realwaste_regime_comparison("cnn", "rand", "paste", five=True),
            "gap_closed": gap_closed_calc("cnn", "rand", "paste"),
        }

    json.dump(summary, open(RESULTS / "summary.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))

    print("Analysis complete (figures: python -m src.s11_figures).")


if __name__ == "__main__":
    main()
