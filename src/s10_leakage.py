"""Duplicate-leakage check (RQ4): does the official file-level TrashNet split inflate accuracy through duplicates?

M1a (colour SVM, tuned on the official validation split) and M2 (seed 0) are trained on the
official train split and scored on the official test split. Test images whose duplicate group
has a member in the official train split are compared with the rest (grouped bootstrap).
Writes results/rq4.json and results/rq4.csv.
"""
import json

import numpy as np
import pandas as pd

from .common import RESULTS
from .features import colour_hist
from .s07_svm import tune_fit
from .s08_cnn import predict, train_model
from .stats import boot_stat, ci, wilson
from .testsets import labels, tn_split


def main():
    (trm, xtr), (vam, xva), (tem, xte) = (tn_split(s, col="official_split") for s in ("train", "val", "test"))
    ytr, yva, yte = labels(trm), labels(vam), labels(tem)
    train_groups = set(trm.group)
    partner = tem.group.isin(train_groups).values
    preds = {}
    _, C, g, m = tune_fit(colour_hist(xtr), ytr, colour_hist(xva), yva)
    preds["M1a colour SVM"] = m.predict(colour_hist(xte))
    net, _, _, _ = train_model(xtr, ytr, xva, yva, lambda k, rng: xtr[k], seed=0)
    preds["M2 ResNet-18 (seed 0)"] = predict(net, xte).argmax(1)
    out, rows = {}, []
    for name, p in preds.items():
        c = (p == yte).astype(float)
        d = boot_stat(lambda w: (w[:, partner] @ c[partner]) / w[:, partner].sum(1) - (w[:, ~partner] @ c[~partner]) / w[:, ~partner].sum(1),
                      tem.group.values, 10_000, seed=7)
        diff = c[partner].mean() - c[~partner].mean()
        lo, hi = ci(d)
        k1, n1, k0, n0 = int(c[partner].sum()), int(partner.sum()), int(c[~partner].sum()), int((~partner).sum())
        rows.append(dict(model=name, n_with_partner=n1, acc_with_partner=k1 / n1, wilson_with=wilson(k1, n1),
                         n_without=n0, acc_without=k0 / n0, wilson_without=wilson(k0, n0), diff=diff, lo=lo, hi=hi,
                         p_one_sided=(1 + (d <= 0).sum()) / (len(d) + 1), official_test_acc=c.mean()))
        out[name] = dict(diff=diff, lo=lo, hi=hi, p=rows[-1]["p_one_sided"])
    pd.DataFrame(rows).to_csv(RESULTS / "rq4.csv", index=False)
    json.dump(out, open(RESULTS / "rq4.json", "w"), indent=1, default=float)
    print(pd.DataFrame(rows).to_string())


if __name__ == "__main__":
    main()
