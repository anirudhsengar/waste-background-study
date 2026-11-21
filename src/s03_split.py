"""Stage 3: one frozen group-aware stratified split.

StratifiedGroupKFold(20, shuffle, seed 0): folds 0-2 test, 3-5 validation, 6-19 train.
The six images in cross-class exact-duplicate pairs are dropped. Also records the official
file-level split (used only for the duplicate-leakage check in s10_leakage).
"""
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .common import DATA, RESULTS, WORK


def main():
    df = pd.read_csv(WORK / "trashnet_groups.csv")
    df["row"] = range(len(df))                      # row index into trashnet_frames.npy
    keep = df[~df.drop_label_conflict].reset_index(drop=True)
    sgkf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=0)
    fold = pd.Series(-1, index=keep.index)
    for k, (_, te) in enumerate(sgkf.split(keep, keep.cls, keep.group)):
        fold.iloc[te] = k
    keep["fold"] = fold.values
    keep["split"] = pd.cut(keep.fold, [-1, 2, 5, 19], labels=["test", "val", "train"]).astype(str)

    official = {}
    for s in ("train", "val", "test"):
        for line in open(DATA / "raw" / f"official_{s}.txt"):
            official[line.split()[0].replace(".jpg", "")] = s
    keep["official_split"] = keep.id.map(official)
    assert keep.official_split.notna().all()
    # every duplicate group stays in one split
    assert keep.groupby("group").split.nunique().max() == 1
    keep.to_csv(WORK / "split.csv", index=False)
    tab = pd.crosstab(keep.cls, keep.split)[["train", "val", "test"]]
    tab.loc["total"] = tab.sum()
    tab.to_csv(RESULTS / "split_counts.csv")
    print(tab)


if __name__ == "__main__":
    main()
