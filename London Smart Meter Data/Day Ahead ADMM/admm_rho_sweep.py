"""Penalty-parameter (rho) sweep for the decentralised ADMM methods.

For each rho it runs the ADMM to (attempted) convergence with ADAPT_RHO off and
records: iterations to converge, wall time, the primal-residual trace, and the
per-building operating-cost RMS vs the central market optimum (both settled at
central_optimisation_showcase.SHARE_TRADE_FEE_FRAC x grid tariff). The rho that
reaches the stopping tolerance in the fewest iterations is the pick.

  python admm_rho_sweep.py                       bilateral, RELAX_BINARIES True and False
  python admm_rho_sweep.py --method exchange     exchange-ADMM (admm_energy_sharing)
  python admm_rho_sweep.py --method both --binaries relax
  python admm_rho_sweep.py --rhos 0.5,1,2,4,8,16 --max-iters 120

Outputs:
  plots/20_admm_rho_sweep.csv    one row per (method, binaries, rho)
  plots/20_admm_rho_sweep.png    primal-residual traces, one panel per (method, binaries)

Honours the scaled-community env vars of central_optimisation_showcase
(SHOWCASE_DATASET, SHOWCASE_PLOTS_SUBDIR, SHOWCASE_SOLVE_TIMELIMIT), so
`SHOWCASE_DATASET=showcase_2013-02-21_N0100 python admm_rho_sweep.py` sweeps at N=100.
Only mutates module globals at runtime — the RHO in the source files is unchanged.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyomo.environ as pyo

import central_optimisation_showcase as C
import admm_bilateral_p2p as BIL
import admm_energy_sharing as EXC
from central_optimisation_showcase import PLOTS_DIR, SHOWCASE_DAY, n_buildings, time_horizon, dt

DEFAULT_RHOS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
FRAC = C.SHARE_TRADE_FEE_FRAC


def _local_cost(subs):
    """(elec_b, gas_b) per building from a list of solved ADMM subproblems (both
    files call the vars p_el / gas)."""
    elec_b = np.array([(C.PRICE * np.array([pyo.value(m.p_el[t]) for t in range(time_horizon)]) * dt).sum()
                       for m in subs])
    gas_b = np.array([(C.gas_price / 100.0 * np.array([pyo.value(m.gas[t]) for t in range(time_horizon)]) * dt).sum()
                      for m in subs])
    return elec_b, gas_b


def _bilateral_metrics(subs, Z):
    elec_b, gas_b = _local_cost(subs)
    sp = FRAC * C.PRICE
    settle_b = np.zeros(len(subs))
    traded = 0.0
    for lo, hi in BIL.PAIRS:
        z = Z[(lo, hi)]
        pay = float((sp * z * dt).sum())
        settle_b[lo] -= pay
        settle_b[hi] += pay
        traded += float(np.abs(z).sum())
    return elec_b + gas_b + settle_b, float((elec_b + gas_b).sum()), traded


def _exchange_metrics(subs, pex):
    elec_b, gas_b = _local_cost(subs)
    settle_b = -np.array([(FRAC * C.PRICE * pex[b] * dt).sum() for b in range(len(subs))])
    return elec_b + gas_b + settle_b, float((elec_b + gas_b).sum()), float(np.maximum(pex, 0).sum())


def _run(method: str, relax: bool, rho: float, max_iters: int, sh_cost_b, sh_op):
    t0 = time.perf_counter()
    if method == "bilateral":
        BIL.RELAX_BINARIES, BIL.RHO, BIL.MAX_ITERS, BIL.ADAPT_RHO = relax, rho, max_iters, False
        subs, _q, _lam, Z, hist = BIL.run_admm()
        r, s, eps = hist[:, 3], hist[:, 4], BIL.EPS_PRIMAL          # bilateral: RMS residuals
        cost_b, op, traded = _bilateral_metrics(subs, Z)
    else:
        EXC.RELAX_BINARIES, EXC.RHO, EXC.MAX_ITERS, EXC.ADAPT_RHO = relax, rho, max_iters, False
        pex, _price, hist, subs = EXC.run_admm()
        r, s, eps = hist[:, 1], hist[:, 2], EXC.EPS_PRIMAL          # exchange: max |Sigma pex| residual
        cost_b, op, traded = _exchange_metrics(subs, pex)
    wall = time.perf_counter() - t0
    n_it = len(hist)
    imin = int(np.argmin(r))
    tail = r[-20:]
    rms = float(np.sqrt(np.mean(((cost_b - sh_cost_b) / np.maximum(np.abs(sh_cost_b), 1e-3)) ** 2)))
    row = dict(
        method=method, binaries="relax" if relax else "exact", rho=rho,
        status="converged" if n_it < max_iters else "MAX_ITERS", iters=n_it,
        min_r=round(float(r[imin]), 6), min_at_iter=imin + 1,
        final_r=round(float(r[-1]), 6), final_s=round(float(s[-1]), 6),
        tail_osc=round(float(tail.std() / max(tail.mean(), 1e-9)), 3),
        wall_s=round(wall, 1), op_cost=round(op, 2), op_gap_vs_central=round(op - sh_op, 3),
        traded_kWh=round(traded, 1), rms_cost_vs_central=round(rms, 4), eps_primal=eps,
    )
    return row, np.asarray(r, float)


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=("bilateral", "exchange", "both"), default="bilateral")
    ap.add_argument("--binaries", choices=("relax", "exact", "both"), default="both")
    ap.add_argument("--rhos", default=",".join(str(x) for x in DEFAULT_RHOS))
    ap.add_argument("--max-iters", type=int, default=200)
    args = ap.parse_args(argv)

    rhos = [float(x) for x in args.rhos.split(",")]
    methods = ["bilateral", "exchange"] if args.method == "both" else [args.method]
    relaxes = {"relax": [True], "exact": [False], "both": [True, False]}[args.binaries]
    combos = [(m, rx) for m in methods for rx in relaxes]

    print(f"rho sweep | day {SHOWCASE_DAY} | N={n_buildings} T={time_horizon} | rhos={rhos} | "
          f"max_iters={args.max_iters}\ncombos: {[(m, 'relax' if r else 'exact') for m, r in combos]}\n")

    C._require_solver()
    sh = C.solve_and_extract(
        C.build_model(True, True, fee_frac=FRAC, fee_mode="market"),
        "central_sh", fee_frac=FRAC, fee_mode="market",
    )
    print(f"central shared op cost GBP {sh['op_cost']:.2f}/day  (reference at frac={FRAC:g})\n")

    rows, traces = [], {}
    for method, relax in combos:
        tag = f"{method}/{'relax' if relax else 'exact'}"
        print(f"===== {tag} =====")
        for rho in rhos:
            try:
                row, r = _run(method, relax, rho, args.max_iters, sh["cost_b"], sh["op_cost"])
            except Exception as e:  # noqa: BLE001
                print(f"  rho {rho}: FAILED ({type(e).__name__}: {e})")
                rows.append(dict(method=method, binaries="relax" if relax else "exact", rho=rho,
                                 status="failed", note=str(e)[:100]))
                continue
            rows.append(row)
            traces[(tag, rho)] = r
            print(f"  rho {rho:>5g}: {row['status']:>9} at {row['iters']:>3} iters, {row['wall_s']:>5}s, "
                  f"final_r {row['final_r']:.2e}, tail_osc {row['tail_osc']:.2f}, "
                  f"RMS vs central {row['rms_cost_vs_central']:.2%}")
        print()

    df = pd.DataFrame(rows)
    out_csv = PLOTS_DIR / "20_admm_rho_sweep.csv"
    df.to_csv(out_csv, index=False)
    print("=" * 118)
    print(df.to_string(index=False))
    print(f"\nsaved {out_csv}")

    conv = df[df["status"] == "converged"] if "status" in df else df.iloc[0:0]
    for (method, bx), g in conv.groupby(["method", "binaries"]):
        best = g.sort_values(["iters", "rms_cost_vs_central"]).iloc[0]
        print(f"  best {method}/{bx}: rho = {best['rho']:g}  ({int(best['iters'])} iters, "
              f"RMS vs central {best['rms_cost_vs_central']:.2%}, op gap GBP {best['op_gap_vs_central']:+.3f}/day)")

    # ---- traces: one panel per combo ----
    ncol = len(combos)
    fig, axes = plt.subplots(1, ncol, figsize=(6.2 * ncol, 5.4), squeeze=False)
    colors = plt.cm.plasma(np.linspace(0.05, 0.9, len(rhos)))
    for ax, (method, relax) in zip(axes[0], combos):
        tag = f"{method}/{'relax' if relax else 'exact'}"
        for c, rho in zip(colors, rhos):
            if (tag, rho) in traces:
                r = traces[(tag, rho)]
                ax.semilogy(range(1, len(r) + 1), r, "-o", ms=2.5, lw=1.3, color=c, label=f"rho={rho:g}")
        eps = BIL.EPS_PRIMAL if method == "bilateral" else EXC.EPS_PRIMAL
        ax.axhline(eps, color="#888", ls=":", lw=1, label=f"EPS_PRIMAL={eps:g}")
        ax.set_xlabel("ADMM iteration")
        ax.set_ylabel("primal residual (kW)" + ("  — RMS" if method == "bilateral" else "  — max|Σpex|"))
        ax.set_title(f"{method}  ·  RELAX_BINARIES={relax}", fontsize=10, fontweight="bold", loc="left")
        ax.grid(True, which="both", color="#eee", lw=0.6)
        ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle(f"{pd.Timestamp(SHOWCASE_DAY):%A %d %b %Y} — ADMM rho sweep, N={n_buildings}",
                 x=0.02, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_png = PLOTS_DIR / "20_admm_rho_sweep.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"saved {out_png}")


if __name__ == "__main__":
    main(sys.argv[1:])
