"""Run the showcase comparison across community sizes and collect the compute cost.

For each N it:
  1. builds  data/showcase_<day>_N<NNNN>.{csv, _batteries.csv}  if missing
     (real LCL ToU households; the diversity filters in sharing_showcase._prep are
     relaxed automatically to reach N — only ~1,100 ToU meters exist, so N=1000 is
     essentially "use nearly all of them"). The showcase DAY is fixed, not re-ranked.
  2. runs showcase_comparison.py in a subprocess with
       SHOWCASE_DATASET          -> that dataset stem
       SHOWCASE_PLOTS_SUBDIR     -> N<NNNN>            (=> every figure in plots/N<NNNN>/)
       SHOWCASE_SOLVE_TIMELIMIT  -> per-solve Gurobi time limit, seconds
     so each size's figures + plots/N<NNNN>/computational_analysis.csv are isolated.
  3. concatenates every per-N computational_analysis.csv into
     plots/scaling_computational_analysis.csv  (adds a `wall_seconds_subprocess` and
     `dataset_build_seconds` column).

The central MIQCP (N² sharing variables) and the bilateral ADMM (N²/2 trade pairs)
do not scale: at N=100 they are slow, at N=1000 they typically run out of memory
while the model is built. showcase_comparison catches those, records status
"failed" in the CSV, and still produces every figure the exchange-ADMM supports.

Usage:
  python run_scaling.py                     # N = 100, 1000
  python run_scaling.py 10 100 1000         # custom sizes (10 uses the existing dataset)
  python run_scaling.py --timelimit 900 --no-admm
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
PLOTS_DIR = HERE / "plots"
_LATEST = DATA_DIR / "showcase_latest.txt"
DAY = _LATEST.read_text().strip() if _LATEST.exists() else "2013-02-21"


def _stem(n: int) -> str:
    return f"showcase_{DAY}_N{n:04d}"


def _ensure_dataset(n: int) -> float:
    """Build the N-building dataset if absent. Returns the build wall time (s)."""
    if n == 10:
        return 0.0  # the default showcase_<day>.csv already is the N=10 community
    if (DATA_DIR / f"{_stem(n)}.csv").exists() and (DATA_DIR / f"{_stem(n)}_batteries.csv").exists():
        print(f"  dataset {_stem(n)} already present")
        return 0.0
    print(f"  building dataset for N={n} ...")
    t0 = time.perf_counter()
    r = subprocess.run([sys.executable, str(HERE / "build_showcase_dataset.py"),
                        "--n", str(n), "--day", DAY])
    dt = time.perf_counter() - t0
    if r.returncode != 0:
        raise RuntimeError(f"build_showcase_dataset.py --n {n} exited {r.returncode}")
    return dt


def main(argv: list[str]) -> None:
    sizes: list[int] = []
    passthrough: list[str] = []
    timelimit = "300"
    it = iter(argv)
    for a in it:
        if a == "--timelimit":
            timelimit = next(it)
        elif a.isdigit():
            sizes.append(int(a))
        else:
            passthrough.append(a)
    if not sizes:
        sizes = [100, 1000]

    all_rows = []
    for n in sizes:
        sub = f"N{n:04d}"
        print(f"\n{'=' * 74}\nN = {n}   ->  plots/{sub}/\n{'=' * 74}")
        try:
            build_s = _ensure_dataset(n)
        except Exception as e:  # noqa: BLE001
            print(f"  ! dataset build failed for N={n}: {e} — skipping")
            continue

        env = dict(os.environ)
        env["SHOWCASE_PLOTS_SUBDIR"] = sub
        env["SHOWCASE_SOLVE_TIMELIMIT"] = timelimit
        if n == 10:
            env.pop("SHOWCASE_DATASET", None)
        else:
            env["SHOWCASE_DATASET"] = _stem(n)

        t0 = time.perf_counter()
        subprocess.run([sys.executable, str(HERE / "showcase_comparison.py"), *passthrough], env=env)
        wall = time.perf_counter() - t0

        csv = PLOTS_DIR / sub / "computational_analysis.csv"
        if csv.exists():
            df = pd.read_csv(csv)
            df["dataset_build_seconds"] = round(build_s, 1)
            df["wall_seconds_subprocess"] = round(wall, 1)
            all_rows.append(df)
        else:
            print(f"  ! {csv} not written")

    if all_rows:
        out = PLOTS_DIR / "scaling_computational_analysis.csv"
        combined = pd.concat(all_rows, ignore_index=True)
        combined.to_csv(out, index=False)
        print(f"\nsaved {out}\n")
        cols = ["n_buildings", "component", "status", "wall_seconds", "iterations",
                "converged", "community_op_cost_gbp", "community_peak_kW", "note"]
        print(combined[[c for c in cols if c in combined.columns]].to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])
