"""Metrics, cluster (duplicate-group) bootstrap, permutation and sign-flip tests, Holm correction."""
import numpy as np
from scipy.stats import binomtest

K = 6


def confusion(y, p, w=None, k=K):
    """Confusion matrix(es). w: None, (n,) or (B, n) image weights -> (k,k) or (B,k,k)."""
    code = np.asarray(y) * k + np.asarray(p)
    onehot = np.zeros((len(code), k * k), np.float32)
    onehot[np.arange(len(code)), code] = 1
    if w is None:
        return onehot.sum(0).reshape(k, k)
    w = np.asarray(w, np.float32)
    out = (np.atleast_2d(w) @ onehot).reshape(-1, k, k)
    return out[0] if w.ndim == 1 else out


def metrics_from_cm(cm):
    """cm: (..., k, k) rows = truth. Returns dict of accuracy, macro_f1, balanced_acc (classes present in truth)."""
    cm = np.asarray(cm, np.float64)
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    rows, cols = cm.sum(-1), cm.sum(-2)
    present = rows > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(2 * tp + (cols - tp) + (rows - tp) > 0, 2 * tp / (2 * tp + (cols - tp) + (rows - tp)), 0.0)
        rec = np.where(present, tp / rows, np.nan)
    n_present = present.sum(-1)
    return {"accuracy": tp.sum(-1) / cm.sum((-2, -1)),
            "macro_f1": (f1 * present).sum(-1) / n_present,
            "balanced_acc": np.nansum(rec, -1) / n_present}


def per_class(cm):
    cm = np.asarray(cm, np.float64)
    tp = np.diag(cm); rows, cols = cm.sum(1), cm.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(cols > 0, tp / cols, 0.0); rec = np.where(rows > 0, tp / rows, 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    return prec, rec, f1


def group_weight_batches(groups, B, seed, chunk=1000):
    """Yield (b, n) image-weight matrices: each row resamples duplicate groups with replacement."""
    _, g = np.unique(groups, return_inverse=True)
    G = g.max() + 1
    rng = np.random.default_rng(seed)
    done = 0
    while done < B:
        b = min(chunk, B - done)
        counts = rng.multinomial(G, np.full(G, 1 / G), size=b).astype(np.float32)   # (b, G)
        yield counts[:, g]
        done += b


def boot_stat(stat, groups, B=2000, seed=0):
    """stat(w) -> (b,) or (b, m) values for a weight matrix w (b, n). Returns stacked draws."""
    return np.concatenate([stat(w) for w in group_weight_batches(groups, B, seed)])


def ci(draws, level=95):
    a = (100 - level) / 2
    return np.percentile(draws, [a, 100 - a], axis=0)


def wmean(x, w):
    """Weighted mean of per-image values x (n,) or (n, m) under weights w (b, n)."""
    return (w @ x) / w.sum(1, keepdims=True) if np.ndim(x) == 2 else (w @ x) / w.sum(1)


def perm_test_balanced_acc(y, p, n=10_000, seed=0, groups=None):
    """Label-permutation null for balanced accuracy against fixed predictions (one-sided).

    When `groups` is supplied, class labels are permuted across duplicate groups (preserving
    within-group label coherence) rather than independently across images.
    """
    y = np.asarray(y)
    p = np.asarray(p)
    obs = metrics_from_cm(confusion(y, p))["balanced_acc"]
    rng = np.random.default_rng(seed)
    if groups is None:
        null = np.array([metrics_from_cm(confusion(rng.permutation(y), p))["balanced_acc"] for _ in range(n)])
    else:
        _, g_inv = np.unique(groups, return_inverse=True)
        # Each duplicate group in split.csv has a single class label (cross-class conflicts dropped).
        g_labels = np.zeros(g_inv.max() + 1, dtype=y.dtype)
        g_labels[g_inv] = y
        null = np.array([
            metrics_from_cm(confusion(rng.permutation(g_labels)[g_inv], p))["balanced_acc"]
            for _ in range(n)
        ])
    return obs, (1 + (null >= obs - 1e-12).sum()) / (n + 1), null


def sign_flip_test(d, groups, n=10_000, seed=0):
    """One-sided paired sign-flip test of mean(d) > 0, flipping whole duplicate groups."""
    _, g = np.unique(groups, return_inverse=True)
    S = np.bincount(g, weights=d)
    obs = S.sum() / len(d)
    rng = np.random.default_rng(seed)
    null = (rng.choice([-1.0, 1.0], size=(n, len(S))) @ S) / len(d)
    return obs, (1 + (null >= obs - 1e-12).sum()) / (n + 1)


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for r, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - r) * p[i]))
        adj[i] = running
    return adj


def mcnemar_exact(correct_a, correct_b):
    """Exact two-sided McNemar test on paired correctness."""
    b = int((correct_a & ~correct_b).sum()); c = int((~correct_a & correct_b).sum())
    return b, c, (binomtest(b, b + c, 0.5).pvalue if b + c else 1.0)


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h
