"""Stage 8: train one neural classifier for one training regime and seed, and predict every evaluation set.

Architectures (`arch`), all ImageNet-pretrained with a new six-way head:
  - `cnn`          : ResNet-18 (IMAGENET1K_V1), conv1-layer3 frozen, layer4 and head trained (M2, primary model)
  - `resnet18_ft`  : ResNet-18 (IMAGENET1K_V1), all layers trained (M2b)
  - `mobilenet_v3` : MobileNetV3-Large (IMAGENET1K_V2), all layers trained (M3)
  - `swin_t`       : Swin-T (IMAGENET1K_V1), all layers trained (M4)

Regimes (`regime`): `orig`, `jitter`, `paste6`, `shuffle6`, `rand6` (paper: T-orig, T-jitter, T-paste, T-shuffle,
T-rand); `paste`, `rand` (four-class U2-Net suite, supplementary); background-only probes `probe_bg6`,
`probe_corners` (paper) and `probe_bg`, `probe_fixed` (supplementary). Training settings are listed in the
README ("Training details").

Writes results/preds/<arch>_<regime>_s<seed>.npz (softmax per evaluation set), one row of results/nn_runs.csv
(s08_version v2) and the per-epoch curve results/logs/<arch>_<regime>_s<seed>.json.

  python -m src.s08_cnn <regime> <seed>            # arch defaults to 'cnn'
  python -m src.s08_cnn <arch> <regime> <seed>
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from sklearn.metrics import f1_score
from torchvision.transforms import v2

from .common import IMAGENET_MEAN, IMAGENET_STD, RESULTS, WORK, seed_everything
from .regimes import Compositor
from .testsets import eval_sets, labels, probe_train_sets, tn_split

torch.set_num_threads(min(10, os.cpu_count() or 1))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"
PRED = RESULTS / "preds"
PRED.mkdir(exist_ok=True)
MEAN = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1).to(DEVICE)
STD = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1).to(DEVICE)
AUG = v2.Compose([v2.RandomResizedCrop(224, scale=(0.7, 1.0), antialias=True), v2.RandomHorizontalFlip()])

ARCHS = ("cnn", "resnet18_ft", "mobilenet_v3", "swin_t")
# Revision of this training script recorded in results/nn_runs.csv. v1 (only the shipped four-class `cnn` runs
# paste, rand, probe_bg, probe_fixed) differed as described in the README ("Determinism").
S08_VERSION = "v2"


class FrozenResNet18(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        r = torchvision.models.resnet18(weights=w)
        self.frozen = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2, r.layer3)
        for p in self.frozen.parameters():
            p.requires_grad = False
        self.layer4, self.pool = r.layer4, r.avgpool
        self.head = nn.Linear(512, 6)

    def train(self, mode: bool = True):
        super().train(mode)
        self.frozen.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.frozen(x)
        return self.head(torch.flatten(self.pool(self.layer4(x)), 1))


def build_model_and_opt(arch: str, pretrained: bool = True, max_epochs: int = 30):
    if arch == "cnn":
        model = FrozenResNet18(pretrained=pretrained).to(DEVICE)
        opt = torch.optim.AdamW(
            [{"params": model.layer4.parameters(), "lr": 3e-4}, {"params": model.head.parameters(), "lr": 1e-3}],
            weight_decay=1e-4,
        )
        sched = None
        return model, opt, sched

    if arch == "resnet18_ft":
        w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = torchvision.models.resnet18(weights=w)
        model.fc = nn.Linear(model.fc.in_features, 6)
        model = model.to(DEVICE)
        early = [p for n, p in model.named_parameters() if not n.startswith(("layer4.", "fc."))]
        opt = torch.optim.AdamW(
            [
                {"params": early, "lr": 1e-4},
                {"params": model.layer4.parameters(), "lr": 3e-4},
                {"params": model.fc.parameters(), "lr": 1e-3},
            ],
            weight_decay=1e-4,
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=1e-5)
        return model, opt, sched

    if arch == "mobilenet_v3":
        w = torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        model = torchvision.models.mobilenet_v3_large(weights=w)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 6)
        model = model.to(DEVICE)
        opt = torch.optim.AdamW(
            [
                {"params": model.features.parameters(), "lr": 2e-4},
                {"params": model.classifier.parameters(), "lr": 1e-3},
            ],
            weight_decay=1e-4,
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=1e-5)
        return model, opt, sched

    if arch == "swin_t":
        w = torchvision.models.Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
        model = torchvision.models.swin_t(weights=w)
        model.head = nn.Linear(model.head.in_features, 6)
        model = model.to(DEVICE)
        backbone = [p for n, p in model.named_parameters() if not n.startswith("head.")]
        opt = torch.optim.AdamW(
            [{"params": backbone, "lr": 1e-4}, {"params": model.head.parameters(), "lr": 5e-4}],
            weight_decay=1e-2,
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=1e-5)
        return model, opt, sched

    raise ValueError(f"Unknown architecture: {arch}")


def to_tensor(batch_u8: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(batch_u8)).permute(0, 3, 1, 2).to(DEVICE, non_blocking=True)


def norm(x_u8: torch.Tensor) -> torch.Tensor:
    return (x_u8.float() / 255.0 - MEAN) / STD


@torch.no_grad()
def predict(model: nn.Module, imgs: np.ndarray, bs: int = 256) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(imgs), bs):
        xb = norm(to_tensor(imgs[i : i + bs]))
        with torch.amp.autocast("cuda", enabled=USE_AMP):
            logits = model(xb)
        out.append(torch.softmax(logits.float(), 1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def train_model(
    xtr: np.ndarray,
    ytr: np.ndarray,
    xva: np.ndarray,
    yva: np.ndarray,
    sample,
    seed: int,
    max_epochs: int = 30,
    arch: str = "cnn",
    pretrained: bool = True,
):
    """Train neural classifier with early stopping (patience 5) on validation macro-F1."""
    seed_everything(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    counts = np.bincount(ytr, minlength=6)
    w = torch.tensor(len(ytr) / (6 * np.maximum(counts, 1)), dtype=torch.float32, device=DEVICE)
    model, opt, sched = build_model_and_opt(arch=arch, pretrained=pretrained, max_epochs=max_epochs)
    lossf = nn.CrossEntropyLoss(weight=w)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    rng = np.random.default_rng(1000 + seed)
    best, best_state, bad, log = -1.0, None, 0, []
    t0 = time.time()
    for ep in range(max_epochs):
        model.train()
        order = rng.permutation(len(ytr))
        last_loss = 0.0
        for i in range(0, len(order), 32):
            idx = order[i : i + 32]
            raw_b = to_tensor(np.stack([sample(int(k), rng) for k in idx]))
            xb = norm(AUG(raw_b))
            yb = torch.from_numpy(ytr[idx]).long().to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                logits = model(xb)
                loss = lossf(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            last_loss = float(loss.item())
        if sched is not None:
            sched.step()
        f1 = float(f1_score(yva, predict(model, xva).argmax(1), average="macro"))
        log.append(dict(epoch=ep, val_macro_f1=f1, last_batch_loss=last_loss, sec=time.time() - t0))
        print(f"[{arch}] {log[-1]}", flush=True)
        if f1 > best:
            best, bad = f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 5:
                break
    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    return model, log, best, best_state


def main(arch: str, regime: str, seed: int) -> None:
    t0 = time.time()
    if regime.startswith("probe_"):
        pkind = regime.split("_", 1)[1]
        (xtr, ytr), (xva, yva) = probe_train_sets(pkind)
        sample = lambda k, rng: xtr[k]
    else:
        meta, xtr = tn_split("train")
        ytr = labels(meta)
        vmeta, xva = tn_split("val")
        yva = labels(vmeta)
        sample = Compositor(meta, xtr, regime)

    model, log, best, best_state = train_model(xtr, ytr, xva, yva, sample, seed, arch=arch)

    out = {}
    if regime.startswith("probe_"):
        pkind = regime.split("_", 1)[1]
        for name, x, _ in eval_sets(include_probe=True, include_v4=True, include_v6=True):
            if pkind == "bg" and name.startswith("var_") and name.endswith("ONLY_BG_T"):
                out[name] = predict(model, x)
            elif pkind == "fixed" and name == "probe_fixed":
                out[name] = predict(model, x)
            elif pkind == "bg6" and name.startswith("var6_") and name.endswith("ONLY_BG_T"):
                out[name] = predict(model, x)
            elif pkind == "corners" and name == "probe_corners":
                out[name] = predict(model, x)
    else:
        for name, x, _ in eval_sets(include_probe=False, include_v4=True, include_v6=True):
            out[name] = predict(model, x)

    pred_file = PRED / f"{arch}_{regime}_s{seed}.npz"
    np.savez_compressed(pred_file, **out)

    row = dict(
        s08_version=S08_VERSION,
        arch=arch,
        regime=regime,
        seed=seed,
        best_val_macro_f1=best,
        epochs=len(log),
        best_epoch=int(np.argmax([l["val_macro_f1"] for l in log])),
        minutes=(time.time() - t0) / 60,
        n_train=len(ytr),
        device=str(DEVICE),
        gpu=torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "",
        torch=torch.__version__,
    )
    path = RESULTS / "nn_runs.csv"
    prev = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if len(prev):
        prev = prev[~((prev.arch == arch) & (prev.regime == regime) & (prev.seed == seed))]
    pd.concat([prev, pd.DataFrame([row])], ignore_index=True).to_csv(path, index=False)
    (RESULTS / "logs").mkdir(exist_ok=True)
    json.dump(log, open(RESULTS / "logs" / f"{arch}_{regime}_s{seed}.json", "w"))


if __name__ == "__main__":
    (RESULTS / "logs").mkdir(exist_ok=True)
    if len(sys.argv) == 3:
        main("cnn", sys.argv[1], int(sys.argv[2]))
    elif len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
    else:
        raise SystemExit("Usage: python -m src.s08_cnn [arch] <regime> <seed>")
