"""Hand-crafted features: HSV colour histogram and SIFT bag of visual words."""
import cv2
import numpy as np
from joblib import Parallel, delayed
from sklearn.cluster import MiniBatchKMeans

from .common import WORK

N_WORDS, MAX_KP, MAX_DESC = 200, 500, 100_000


def colour_hist(imgs):
    """8 x 8 x 8 HSV histogram over OpenCV ranges H 0-180, S 0-256, V 0-256, L1-normalised."""
    out = np.empty((len(imgs), 512), np.float32)
    for i, im in enumerate(imgs):
        hsv = cv2.cvtColor(np.ascontiguousarray(im), cv2.COLOR_RGB2HSV)
        h = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256]).ravel()
        out[i] = h / max(h.sum(), 1)
    return out


def _sift_chunk(imgs):
    sift = cv2.SIFT_create(nfeatures=MAX_KP)
    res = []
    for im in imgs:
        _, d = sift.detectAndCompute(cv2.cvtColor(np.ascontiguousarray(im), cv2.COLOR_RGB2GRAY), None)
        res.append(np.zeros((0, 128), np.float32) if d is None else d[:MAX_KP])
    return res


def sift_descriptors(imgs, n_jobs=10):
    chunks = np.array_split(np.arange(len(imgs)), max(1, min(n_jobs * 4, len(imgs))))
    parts = Parallel(n_jobs=n_jobs)(delayed(_sift_chunk)(np.asarray(imgs[c])) for c in chunks if len(c))
    return [d for p in parts for d in p]


def fit_codebook(train_imgs):
    """Fitted once on <= 100k descriptors from T-orig training images (seed 0), reused everywhere."""
    path = WORK / "sift_codebook.npy"
    if path.exists():
        return np.load(path)
    desc = np.concatenate(sift_descriptors(train_imgs))
    rng = np.random.default_rng(0)
    if len(desc) > MAX_DESC:
        desc = desc[rng.choice(len(desc), MAX_DESC, replace=False)]
    km = MiniBatchKMeans(N_WORDS, random_state=0, batch_size=4096, n_init=3).fit(desc)
    np.save(path, km.cluster_centers_.astype(np.float32))
    return km.cluster_centers_.astype(np.float32)


def bovw(imgs, centers):
    """Hard-assignment histogram, L1-normalised then square-rooted; no keypoints -> zero vector."""
    c2 = (centers ** 2).sum(1)
    out = np.zeros((len(imgs), len(centers)), np.float32)
    for i, d in enumerate(sift_descriptors(imgs)):
        if len(d):
            words = np.argmin(c2[None] - 2 * d @ centers.T, axis=1)
            h = np.bincount(words, minlength=len(centers)).astype(np.float32)
            out[i] = np.sqrt(h / h.sum())
    return out
