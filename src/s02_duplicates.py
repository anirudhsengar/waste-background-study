"""Stage 2: duplicate audit before any split.

Links: (a) identical decoded pixels (MD5); (b) pHash AND dHash Hamming distance <= 6;
(c) pose variants: ResNet-50 cosine >= 0.90 across all classes, each candidate pair judged on contact sheets
(stop after 20 consecutive pairs judged different). The recorded verdicts in configs/duplicate_review.csv were
made by an AI model (Claude) with labels hidden; the `reviewer` column says so.

Usage:
  python -m src.s02_duplicates candidates   # writes candidate pairs + review contact sheets
  python -m src.s02_duplicates groups       # reads configs/duplicate_review.csv, writes groups
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision

from .common import CLASSES, DATA, FIGS, IMAGENET_MEAN, IMAGENET_STD, RESULTS, ROOT, WORK

REVIEW = ROOT / "configs" / "duplicate_review.csv"
HASH_T = 6
COS_T = 0.90


def hamming_matrix(hex_hashes):
    bits = np.array([[int(b) for b in bin(int(h, 16))[2:].zfill(64)] for h in hex_hashes], dtype=np.uint8)
    b = bits.astype(np.int32)
    return (b[:, None, :] != b[None, :, :]).sum(-1) if len(b) < 600 else _hamming_blocked(b)


def _hamming_blocked(b):
    n = len(b)
    out = np.empty((n, n), dtype=np.uint8)
    ones = b.sum(1)
    inner = b @ b.T                       # matching ones
    out[:] = (ones[:, None] + ones[None, :] - 2 * inner).astype(np.uint8)
    return out


@torch.no_grad()
def embed(frames, batch=64):
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = torch.nn.Identity()
    model.eval()
    out = []
    for i in range(0, len(frames), batch):
        x = torch.from_numpy(((frames[i:i + batch] / 255.0 - IMAGENET_MEAN) / IMAGENET_STD).astype(np.float32))
        out.append(model(x.permute(0, 3, 1, 2)).numpy())
    e = np.concatenate(out)
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def hash_pairs(df):
    """Exact (MD5) and perceptual (pHash AND dHash <= 6) pairs, i < j."""
    pairs = []
    for _, g in df.groupby("md5"):
        idx = g.index.tolist()
        pairs += [(a, b, "md5") for k, a in enumerate(idx) for b in idx[k + 1:]]
    ph, dh = _hamming_blocked(_bits(df.phash)), _bits_h(df.dhash)
    iu = np.triu_indices(len(df), 1)
    near = (ph[iu] <= HASH_T) & (dh[iu] <= HASH_T)
    exact = {(a, b) for a, b, _ in pairs}
    pairs += [(a, b, "hash") for a, b in zip(iu[0][near], iu[1][near]) if (a, b) not in exact]
    return pairs


def _bits(hexes):
    return np.array([[int(b) for b in bin(int(h, 16))[2:].zfill(64)] for h in hexes], dtype=np.int32)


def _bits_h(hexes):
    return _hamming_blocked(_bits(hexes))


def contact_sheet(frames, df, pairs, out, title):
    n = len(pairs)
    fig, axes = plt.subplots(n, 2, figsize=(4.2, 2.1 * n))
    axes = np.atleast_2d(axes)
    for r, (a, b, s) in enumerate(pairs):
        for k, i in enumerate((a, b)):
            axes[r, k].imshow(frames[i]); axes[r, k].set_xticks([]); axes[r, k].set_yticks([])
        axes[r, 0].set_ylabel(f"#{r}  cos {s:.3f}", fontsize=8)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=80); plt.close(fig)


def candidates(name):
    df = pd.read_csv(WORK / f"{name}_manifest.csv")
    frames = np.load(WORK / f"{name}_frames.npy", mmap_mode="r")
    hp = hash_pairs(df)
    pd.DataFrame(hp, columns=["i", "j", "link"]).assign(id_i=lambda d: df.id.values[d.i], id_j=lambda d: df.id.values[d.j],
                                                         cls_i=lambda d: df.cls.values[d.i], cls_j=lambda d: df.cls.values[d.j]
                                                         ).to_csv(WORK / f"{name}_hash_pairs.csv", index=False)
    print(name, "hash pairs:", pd.Series([p[2] for p in hp]).value_counts().to_dict())
    if name != "trashnet":
        return
    e = embed(np.asarray(frames))
    np.save(WORK / "trashnet_resnet50_emb.npy", e)
    sim = e @ e.T
    iu = np.triu_indices(len(df), 1)
    keep = sim[iu] >= COS_T
    linked = {(a, b) for a, b, _ in hp}
    cand = sorted([(a, b, sim[a, b]) for a, b in zip(iu[0][keep], iu[1][keep]) if (a, b) not in linked], key=lambda t: -t[2])
    cdf = pd.DataFrame(cand, columns=["i", "j", "cos"])
    cdf["id_i"], cdf["id_j"] = df.id.values[cdf.i], df.id.values[cdf.j]
    cdf["rank"] = range(len(cdf))
    cdf.to_csv(WORK / "trashnet_embedding_candidates.csv", index=False)
    print("embedding candidates (cos >= 0.90, not hash-linked):", len(cdf))
    sheet_dir = WORK / "dup_review"; sheet_dir.mkdir(exist_ok=True)
    for s in range(0, min(len(cand), 200), 20):          # first 200 in similarity order; review stops earlier
        contact_sheet(frames, df, cand[s:s + 20], sheet_dir / f"cand_{s:03d}.png", f"candidates {s}-{s + 19} (labels hidden)")
    hp_sheet = [(a, b, 1.0) for a, b, _ in hp]
    for s in range(0, len(hp_sheet), 20):
        contact_sheet(frames, df, hp_sheet[s:s + 20], sheet_dir / f"hash_{s:03d}.png", f"hash-linked pairs {s}-{s + 19}")


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def groups():
    out = {}
    for name in ("trashnet", "realwaste"):
        df = pd.read_csv(WORK / f"{name}_manifest.csv")
        hp = pd.read_csv(WORK / f"{name}_hash_pairs.csv")
        uf = UnionFind(len(df))
        for a, b in zip(hp.i, hp.j):
            uf.union(a, b)
        n_pose = 0
        if name == "trashnet":
            rev = pd.read_csv(REVIEW)
            dup = rev[rev.decision == "duplicate"]
            for a, b in zip(dup.i, dup.j):
                uf.union(a, b)
            n_pose = len(dup)
        roots = [uf.find(i) for i in range(len(df))]
        df["group"] = pd.factorize(pd.Series(roots))[0]
        # cross-class exact duplicates: label conflicts, both copies dropped
        md5 = hp[hp.link == "md5"]
        conflict = md5[md5.cls_i != md5.cls_j]
        df["drop_label_conflict"] = df.id.isin(set(conflict.id_i) | set(conflict.id_j))
        df.to_csv(WORK / f"{name}_groups.csv", index=False)
        sizes = df.group.value_counts()
        out[name] = dict(images=len(df), md5_pairs=int((hp.link == "md5").sum()), hash_pairs=int((hp.link == "hash").sum()),
                         pose_pairs_confirmed=n_pose, dup_groups=int((sizes > 1).sum()), images_in_dup_groups=int(sizes[sizes > 1].sum()),
                         cross_class_exact_pairs=len(conflict), dropped_for_label_conflict=int(df.drop_label_conflict.sum()))
        if len(conflict):
            print(name, "cross-class exact duplicates:", list(zip(conflict.id_i, conflict.id_j)))
    # agreement with Prayitno & Putra's released TrashNet groups
    pp = pd.read_csv(DATA / "raw" / "prayitno" / "tn_index.csv")
    pp["group"] = np.load(DATA / "raw" / "prayitno" / "tn_groups.npy", allow_pickle=True)
    pp["id"] = pp.path.str.split("/").str[1].str.replace(".jpg", "", regex=False)
    tn = pd.read_csv(WORK / "trashnet_groups.csv").merge(pp[["id", "group"]], on="id", suffixes=("", "_pp"))
    ours = {frozenset(g.id) for _, g in tn.groupby("group") if len(g) > 1}
    theirs = {frozenset(g.id) for _, g in tn.groupby("group_pp") if len(g) > 1}
    pair = lambda S: {frozenset((a, b)) for g in S for a in g for b in g if a < b}
    po, pt = pair(ours), pair(theirs)
    out["agreement_with_prayitno_putra"] = dict(our_groups=len(ours), their_groups=len(theirs), identical_groups=len(ours & theirs),
                                                our_pairs=len(po), their_pairs=len(pt), shared_pairs=len(po & pt))
    pd.DataFrame(out).T.to_csv(RESULTS / "duplicate_audit.csv")
    print(pd.Series(out).to_string())


if __name__ == "__main__":
    if sys.argv[1] == "candidates":
        candidates("trashnet"); candidates("realwaste")
    else:
        groups()
