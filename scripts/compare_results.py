"""Compare regenerated results/*.csv and results/*.json with reference versions.

The reference is a directory of copies taken before regeneration (--ref-dir, as scripts/reproduce_stats.sh does)
or a git revision (--ref, default HEAD). Numbers may differ only by floating-point noise (relative tolerance 1e-9,
absolute 1e-12). Exits 1 on a mismatch.

  python scripts/compare_results.py [--ref REV | --ref-dir DIR]
"""
from __future__ import annotations

import io
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "performance.csv", "performance_seed_stats.csv", "reliance.csv", "reliance_per_class.csv", "reliance6.csv",
    "reliance6_per_class.csv", "probes.csv", "mask_quality_covariate.csv", "error_modes.csv", "summary.json",
    "contrasts.json", "regime_contrasts.csv", "seed_variability.csv", "exclusion_reasons.csv",
    "bggap_sensitivity.csv", "sorting_errors.csv", "error_cost.csv",
]
RTOL, ATOL = 1e-9, 1e-12


def committed(path: str, ref: str) -> str:
    return subprocess.run(["git", "show", f"{ref}:{path}"], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def reference(name: str, ref: str, ref_dir: Path | None) -> str:
    return (ref_dir / name).read_text() if ref_dir else committed(f"results/{name}", ref)


def same_json(a, b) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same_json(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same_json(x, y) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return (math.isnan(a) and math.isnan(b)) or math.isclose(a, b, rel_tol=RTOL, abs_tol=ATOL)
    return a == b


def same_csv(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    if list(a.columns) != list(b.columns) or a.shape != b.shape:
        return False
    for c in a.columns:
        x, y = a[c], b[c]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            if not np.allclose(x.to_numpy(float), y.to_numpy(float), rtol=RTOL, atol=ATOL, equal_nan=True):
                return False
        elif not x.astype(str).equals(y.astype(str)):
            return False
    return True


def main() -> None:
    ref = sys.argv[sys.argv.index("--ref") + 1] if "--ref" in sys.argv else "HEAD"
    ref_dir = Path(sys.argv[sys.argv.index("--ref-dir") + 1]) if "--ref-dir" in sys.argv else None
    bad = []
    for name in FILES:
        path = f"results/{name}"
        old = reference(name, ref, ref_dir)
        new = (ROOT / path).read_text()
        if name.endswith(".json"):
            ok = same_json(json.loads(new), json.loads(old))
        else:
            ok = same_csv(pd.read_csv(io.StringIO(new)), pd.read_csv(io.StringIO(old)))
        print(f"{'OK  ' if ok else 'DIFF'} {path}")
        if not ok:
            bad.append(path)
    print("all regenerated tables match the reference" if not bad else f"{len(bad)} file(s) differ")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
