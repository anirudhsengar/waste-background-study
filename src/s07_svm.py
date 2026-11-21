"""Stage 7: majority baseline (M0) and RBF SVMs on colour histograms (M1a) and SIFT bag of visual words (M1b).

For each feature: C and gamma chosen on validation macro-F1 from C in {0.1, 1, 10, 100} x gamma in {scale, 1e-3,
1e-2, 0.1} (balanced class weights) for the regimes orig, jitter, paste6, shuffle6, rand6 (paper) and paste, rand
(four-class, supplementary), each augmented regime with three paste draws; and for the background-only probes
bg6, corners (paper) and bg, fixed (supplementary).

Writes results/preds/svm_<feat>_<regime>_d<draw>.npz (predicted classes per evaluation set and the RealWaste
decision values rw_dec; probe files use the keys probe, probe_erode3, probe_dilate3), results/preds/m0_orig_d0.npz,
results/svm_tuning.csv and results/m0.json.

  python -m src.s07_svm
"""
import itertools
import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.svm import SVC

from .common import RESULTS
from .features import bovw, colour_hist, fit_codebook
from .regimes import Compositor
from .testsets import eval_sets, labels, probe_train_sets, tn_split

PRED = RESULTS / "preds"
PRED.mkdir(exist_ok=True)
GRID = list(itertools.product([0.1, 1, 10, 100], ["scale", 0.001, 0.01, 0.1]))


def tune_fit(Xtr, ytr, Xva, yva):
    best = None
    for C, g in GRID:
        m = SVC(C=C, gamma=g, kernel="rbf", class_weight="balanced").fit(Xtr, ytr)
        f = f1_score(yva, m.predict(Xva), average="macro")
        if best is None or f > best[0]:
            best = (f, C, g, m)
    return best


def main():
    tr_meta, tr_x = tn_split("train")
    ytr = labels(tr_meta)
    centers = fit_codebook(tr_x)
    feats = {"hist": colour_hist, "sift": lambda x: bovw(x, centers)}
    sets = list(eval_sets(include_probe=True, include_v4=True, include_v6=True))
    tuning = []
    for fname, f in feats.items():
        print(f"Extracting {fname} features across {len(sets)} eval sets...", flush=True)
        F = {name: f(x) for name, x, _ in sets}
        Y = {name: y for name, _, y in sets}
        Ftr = f(tr_x)
        runs = [("orig", 0, Ftr, ytr)]
        for regime in ("jitter", "paste", "rand", "paste6", "rand6", "shuffle6"):
            comp = Compositor(tr_meta, tr_x, regime)
            for d in range(3):
                idx, xs = comp.one_copy_each(seed=d)
                runs.append((regime, d, np.concatenate([Ftr, f(xs)]), np.concatenate([ytr, ytr[idx]])))
        for regime, d, X, y in runs:
            score, C, g, m = tune_fit(X, y, F["tn_val"], Y["tn_val"])
            tuning.append(dict(feat=fname, regime=regime, draw=d, C=C, gamma=g, val_macro_f1=score, n_train=len(y)))
            np.savez_compressed(
                PRED / f"svm_{fname}_{regime}_d{d}.npz",
                rw_dec=m.decision_function(F["rw"]),
                **{n: m.predict(F[n]) for n in F if not n.startswith("probe")},
            )
            print(tuning[-1], flush=True)

        for kind in ("bg", "fixed", "bg6", "corners"):
            (xa, ya), (xv, yv) = probe_train_sets(kind)
            score, C, g, m = tune_fit(f(xa), ya, f(xv), yv)
            tuning.append(dict(feat=fname, regime=f"probe_{kind}", draw=0, C=C, gamma=g, val_macro_f1=score, n_train=len(ya)))
            if kind == "bg":
                out = {"probe": m.predict(F["var_base_ONLY_BG_T"])}
                out.update({f"probe_{k}": m.predict(F[f"var_{k}_ONLY_BG_T"]) for k in ("erode3", "dilate3")})
            elif kind == "fixed":
                out = {"probe": m.predict(F["probe_fixed"])}
            elif kind == "bg6":
                out = {"probe": m.predict(F["var6_base_ONLY_BG_T"])}
                out.update({f"probe_{k}": m.predict(F[f"var6_{k}_ONLY_BG_T"]) for k in ("erode3", "dilate3")})
            elif kind == "corners":
                out = {"probe": m.predict(F["probe_corners"])}
            np.savez_compressed(PRED / f"svm_{fname}_probe_{kind}_d0.npz", **out)
            print(tuning[-1], flush=True)

    pd.DataFrame(tuning).to_csv(RESULTS / "svm_tuning.csv", index=False)
    maj = int(np.bincount(ytr).argmax())
    np.savez_compressed(
        PRED / "m0_orig_d0.npz",
        **{n: np.full(len(y), maj) for n, _, y in sets if not n.startswith("probe")},
    )
    json.dump({"majority_class_index": maj}, open(RESULTS / "m0.json", "w"))


if __name__ == "__main__":
    main()
