"""Sweep the P2P settlement-price fraction (default 0.1 -> 1.0 x grid tariff) for
the central market model and the bilateral consensus-ADMM.

In FEE_MODE="market" the settlement is a pure buyer<->seller transfer: for ANY
feasible routing, Sum_b(recv_b(t) - sent_b(t)) = 0 identically at every t (every
kWh sent by one building is received by another), so the trade term contributes
EXACTLY ZERO to the objective for every fee_frac -- the optimal physical dispatch
is therefore invariant to the settlement fraction. Same argument for the
bilateral ADMM's fixed-price settlement. So this sweep needs only ONE central
solve and ONE ADMM run: every fraction is then just a cheap re-settlement of the
same dispatch, not a re-optimisation.

For each frac it recomputes every building's operating cost and net benefit vs
the own-battery baseline (no sharing), for both central and bilateral, and
tracks whether the split stays individually rational (every building's benefit
>= 0) -- the participation frontier for a P2P market design. It also reports the
COMMUNITY-level impact: total benefit from sharing is fixed (a pure transfer only
moves it between buyer and seller, so the total is flat vs frac) and how it
compares in size to the no_battery / own_battery baselines.

Outputs: plots/21_price_fraction_sweep.{png,csv}

  python price_fraction_sweep.py                  # frac = 0.1, 0.2, ..., 1.0
  python price_fraction_sweep.py --step 0.05       # finer grid
  python price_fraction_sweep.py --fracs 0.1,0.5,1.0
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pyomo.environ as pyo

import central_optimisation_showcase as C
import admm_bilateral_p2p as BIL
from central_optimisation_showcase import PLOTS_DIR, SHOWCASE_DAY, BUILDING_IDS, ASSETS, PRICE, dt

_ASSET_COLOR = {"battery+pv": "#2e8b6e", "battery": "#7b4bc9", "pv": "#e0a800", "none": "#8a8f98"}


def _bilateral_physical(subs, Z):
    T = BIL.T
    elec_b = np.array([(PRICE * np.array([pyo.value(m.p_el[t]) for t in range(T)]) * dt).sum() for m in subs])
    gas_b = np.array(
        [(C.gas_price / 100.0 * np.array([pyo.value(m.gas[t]) for t in range(T)]) * dt).sum() for m in subs]
    )
    flow = {b: np.zeros((BIL.N - 1, T)) for b in range(BIL.N)}
    for lo, hi in BIL.PAIRS:
        z = Z[(lo, hi)]
        flow[lo][BIL.NB[lo].index(hi)] = z
        flow[hi][BIL.NB[hi].index(lo)] = -z
    return elec_b, gas_b, flow


def _bilateral_at_frac(elec_b, gas_b, Z, frac):
    settle_price = frac * PRICE
    settle_b = np.zeros(BIL.N)
    for lo, hi in BIL.PAIRS:
        pay = float((settle_price * Z[(lo, hi)] * dt).sum())
        settle_b[lo] -= pay
        settle_b[hi] += pay
    return elec_b + gas_b + settle_b, settle_b


def _central_at_frac(elec_b, gas_b, sent, recv, frac):
    seller_fee_b = (frac * PRICE[None, :] * sent * dt).sum(axis=1)
    buyer_fee_b = (frac * PRICE[None, :] * recv * dt).sum(axis=1)
    trade_cost_b = buyer_fee_b - seller_fee_b
    return elec_b + gas_b + trade_cost_b, trade_cost_b, seller_fee_b


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", type=float, default=0.1)
    ap.add_argument("--fracs", default=None, help="comma-separated list, overrides --step")
    args = ap.parse_args(argv)
    fracs = (
        np.array([float(x) for x in args.fracs.split(",")])
        if args.fracs
        else np.round(np.arange(0.1, 1.0 + 1e-9, args.step), 3)
    )

    print(f"day {SHOWCASE_DAY} | N={C.n_buildings} | fracs = {list(fracs)}\n")
    C._require_solver()

    nb = C.solve_and_extract(C.build_model(False, False), "no_battery")
    ob = C.solve_and_extract(C.build_model(True, False), "own_battery")
    sh = C.solve_and_extract(
        C.build_model(True, True, fee_frac=0.5, fee_mode="market"), "shared", fee_frac=0.5, fee_mode="market"
    )
    print(
        f"central: no_battery £{nb['op_cost']:.2f}/day, own_battery £{ob['op_cost']:.2f}/day, "
        f"shared elec+gas £{(sh['elec_b'] + sh['gas_b']).sum():.2f}/day (frac-invariant in market mode)\n"
    )

    print("running bilateral ADMM once ...")
    subs, _q, _lam, Z, _hist = BIL.run_admm()
    b_elec, b_gas, flow = _bilateral_physical(subs, Z)
    print(f"bilateral: elec+gas £{(b_elec + b_gas).sum():.2f}/day\n")

    def role(sent_kwh, recv_kwh):
        if sent_kwh > recv_kwh + 0.1:
            return "seller"
        if recv_kwh > sent_kwh + 0.1:
            return "buyer"
        return "-"

    c_role = [role(s, r) for s, r in zip(sh["sent"].sum(axis=1), sh["recv"].sum(axis=1))]
    b_sent = np.array([np.maximum(flow[b], 0).sum() for b in range(BIL.N)])
    b_recv = np.array([np.maximum(-flow[b], 0).sum() for b in range(BIL.N)])
    b_role = [role(s, r) for s, r in zip(b_sent, b_recv)]

    def role_totals(roles, benefit):
        buyer = float(sum(v for r, v in zip(roles, benefit) if r == "buyer"))
        seller = float(sum(v for r, v in zip(roles, benefit) if r == "seller"))
        other = float(sum(v for r, v in zip(roles, benefit) if r == "-"))
        return buyer, seller, other

    rows, summary = [], []
    for frac in fracs:
        c_cost, c_trade, c_seller_fee = _central_at_frac(sh["elec_b"], sh["gas_b"], sh["sent"], sh["recv"], frac)
        b_cost, b_settle = _bilateral_at_frac(b_elec, b_gas, Z, frac)
        c_benefit = ob["cost_b"] - c_cost
        b_benefit = ob["cost_b"] - b_cost
        for i, bid in enumerate(BUILDING_IDS):
            rows.append(
                dict(
                    method="central",
                    frac=frac,
                    LCLid=bid,
                    assets=ASSETS[i],
                    role=c_role[i],
                    cost=round(c_cost[i], 3),
                    net_benefit=round(c_benefit[i], 3),
                )
            )
            rows.append(
                dict(
                    method="bilateral",
                    frac=frac,
                    LCLid=bid,
                    assets=ASSETS[i],
                    role=b_role[i],
                    cost=round(b_cost[i], 3),
                    net_benefit=round(b_benefit[i], 3),
                )
            )
        c_buy, c_sell, c_oth = role_totals(c_role, c_benefit)
        b_buy, b_sell, b_oth = role_totals(b_role, b_benefit)
        summary.append(
            dict(
                frac=frac,
                central_min_benefit=round(float(c_benefit.min()), 3),
                central_all_IR=bool(c_benefit.min() >= -1e-6),
                central_transfer=round(float(c_seller_fee.sum()), 3),
                central_community_benefit=round(float(c_benefit.sum()), 3),
                central_buyer_total=round(c_buy, 3),
                central_seller_total=round(c_sell, 3),
                central_other_total=round(c_oth, 3),
                bilateral_min_benefit=round(float(b_benefit.min()), 3),
                bilateral_all_IR=bool(b_benefit.min() >= -1e-6),
                bilateral_transfer=round(float(np.maximum(b_settle, 0).sum()), 3),
                bilateral_community_benefit=round(float(b_benefit.sum()), 3),
                bilateral_buyer_total=round(b_buy, 3),
                bilateral_seller_total=round(b_sell, 3),
                bilateral_other_total=round(b_oth, 3),
            )
        )
        print(
            f"  frac {frac:.2f}: central min-benefit £{c_benefit.min():+.3f} "
            f"({'IR ok' if c_benefit.min() >= -1e-6 else 'IR VIOLATED'}), community £{c_benefit.sum():+.3f}/day | "
            f"bilateral min-benefit £{b_benefit.min():+.3f} "
            f"({'IR ok' if b_benefit.min() >= -1e-6 else 'IR VIOLATED'}), community £{b_benefit.sum():+.3f}/day"
        )

    batt_benefit = nb["op_cost"] - ob["op_cost"]

    df = pd.DataFrame(rows)
    sdf = pd.DataFrame(summary)
    out_csv = PLOTS_DIR / "21_price_fraction_sweep.csv"
    df.to_csv(out_csv, index=False)
    sdf.to_csv(PLOTS_DIR / "21_price_fraction_sweep_summary.csv", index=False)
    print(f"\nsaved {out_csv}")
    print(sdf.to_string(index=False))

    # ---------------------------------------------------------------- plot
    fig, ((axC, axB, axRoleC, axRoleB), (axIR, axT, axCB, axBar)) = plt.subplots(2, 4, figsize=(24, 9.5))
    _role_ls = {"seller": "-", "buyer": "--", "-": ":"}
    for ax, method, roles in ((axC, "central", c_role), (axB, "bilateral", b_role)):
        sub = df[df["method"] == method]
        for i, bid in enumerate(BUILDING_IDS):
            b = sub[sub["LCLid"] == bid]
            ax.plot(
                b["frac"], b["net_benefit"], "-o", ms=3, lw=1.3, color=_ASSET_COLOR[ASSETS[i]], ls=_role_ls[roles[i]]
            )
        ax.axhline(0, color="#888", lw=0.8)
        ax.set_xlabel("settlement price fraction  (x grid tariff)")
        ax.set_ylabel("net benefit vs own-battery (£/day)")
        ax.set_title(f"{method}: per-building benefit vs settlement price", fontsize=10, fontweight="bold", loc="left")
        ax.grid(True, color="#eee", lw=0.6)
        asset_handles = [Line2D([0], [0], color=c, lw=2) for c in _ASSET_COLOR.values()]
        role_handles = [Line2D([0], [0], color="#333", ls=ls) for ls in _role_ls.values()]
        leg1 = ax.legend(
            asset_handles,
            _ASSET_COLOR.keys(),
            frameon=False,
            fontsize=7,
            loc="upper left" if method == "central" else "lower right",
            title="asset",
            title_fontsize=7,
        )
        ax.add_artist(leg1)
        ax.legend(
            role_handles,
            [f"{r} (line style)" for r in _role_ls],
            frameon=False,
            fontsize=7,
            loc="upper right" if method == "central" else "upper left",
        )

    # ---- COMMUNITY CHANGES WITH FRAC: how the (fixed) total splits between buyers and sellers ----
    for ax, prefix, method in ((axRoleC, "central", "central"), (axRoleB, "bilateral", "bilateral")):
        ax.plot(sdf["frac"], sdf[f"{prefix}_buyer_total"], "-o", color="#3b6bb0", label="buyers (community total)")
        ax.plot(sdf["frac"], sdf[f"{prefix}_seller_total"], "-o", color="#c0392b", label="sellers (community total)")
        if sdf[f"{prefix}_other_total"].abs().max() > 1e-3:
            ax.plot(sdf["frac"], sdf[f"{prefix}_other_total"], "-o", color="#8a8f98", label="non-participants (total)")
        ax.plot(
            sdf["frac"], sdf[f"{prefix}_community_benefit"], ":", color="#333", lw=1.3, label="community total (fixed)"
        )
        ax.axhline(0, color="#888", lw=0.8)
        ax.set_xlabel("settlement price fraction  (x grid tariff)")
        ax.set_ylabel("£ / day (aggregated across buildings)")
        ax.set_title(
            f"{method}: community split by role\n(who captures the fixed benefit)",
            fontsize=10,
            fontweight="bold",
            loc="left",
        )
        ax.grid(True, color="#eee", lw=0.6)
        ax.legend(frameon=False, fontsize=7.5)

    axIR.plot(sdf["frac"], sdf["central_min_benefit"], "-o", color="#3b6bb0", label="central (worst-off building)")
    axIR.plot(
        sdf["frac"], sdf["bilateral_min_benefit"], "-o", color="#c0392b", label="bilateral ADMM (worst-off building)"
    )
    axIR.axhline(0, color="#888", lw=1, ls=":")
    _all_min = pd.concat([sdf["central_min_benefit"], sdf["bilateral_min_benefit"]])
    _lo, _hi = float(_all_min.min()), float(_all_min.max())
    _pad = 0.15 * max(_hi - _lo, 0.05)
    axIR.set_ylim(_lo - _pad, _hi + _pad)
    axIR.fill_between(sdf["frac"], _lo - _pad, 0, color="#c0392b", alpha=0.08)
    axIR.set_xlabel("settlement price fraction  (x grid tariff)")
    axIR.set_ylabel("min net benefit across buildings (£/day)")
    axIR.set_title(
        "Individual-rationality frontier\n(shaded = someone worse off than not sharing)",
        fontsize=10,
        fontweight="bold",
        loc="left",
    )
    axIR.grid(True, color="#eee", lw=0.6)
    axIR.legend(frameon=False, fontsize=8)

    axT.plot(sdf["frac"], sdf["central_transfer"], "-o", color="#3b6bb0", label="central: £ moved buyer→seller")
    axT.plot(sdf["frac"], sdf["bilateral_transfer"], "-o", color="#c0392b", label="bilateral: £ moved buyer→seller")
    axT.set_xlabel("settlement price fraction  (x grid tariff)")
    axT.set_ylabel("£ / day transferred")
    axT.set_title("Settlement volume\n(redistribution grows with frac)", fontsize=10, fontweight="bold", loc="left")
    axT.grid(True, color="#eee", lw=0.6)
    axT.legend(frameon=False, fontsize=8)

    # ---- COMMUNITY IMPACT: total benefit from sharing is a fixed pie, not moved by frac ----
    axCB.plot(sdf["frac"], sdf["central_community_benefit"], "-o", color="#3b6bb0", label="central (community total)")
    axCB.plot(
        sdf["frac"],
        sdf["bilateral_community_benefit"],
        "-o",
        color="#c0392b",
        label="bilateral ADMM (community total)",
    )
    c_flat = float(sdf["central_community_benefit"].mean())
    b_flat = float(sdf["bilateral_community_benefit"].mean())
    axCB.axhline(c_flat, color="#3b6bb0", ls=":", lw=0.8)
    axCB.axhline(b_flat, color="#c0392b", ls=":", lw=0.8)
    axCB.annotate(
        f"£{c_flat:.2f}/day",
        (sdf["frac"].iloc[-1], c_flat),
        xytext=(4, 6),
        textcoords="offset points",
        fontsize=8,
        color="#3b6bb0",
    )
    axCB.annotate(
        f"£{b_flat:.2f}/day",
        (sdf["frac"].iloc[-1], b_flat),
        xytext=(4, -12),
        textcoords="offset points",
        fontsize=8,
        color="#c0392b",
    )
    _span = max(abs(c_flat), abs(b_flat), 0.5)
    axCB.set_ylim(min(0, c_flat, b_flat) - 0.3 * _span, max(c_flat, b_flat) + 0.3 * _span)
    axCB.set_xlabel("settlement price fraction  (x grid tariff)")
    axCB.set_ylabel("community net benefit vs own-battery (£/day)")
    axCB.set_title(
        "Community impact: TOTAL benefit is fixed\n(frac only redistributes it — see IR frontier)",
        fontsize=10,
        fontweight="bold",
        loc="left",
    )
    axCB.grid(True, color="#eee", lw=0.6)
    axCB.legend(frameon=False, fontsize=8, loc="center left")

    # ---- COMMUNITY IMPACT: how big is that fixed pie vs the no_battery / own_battery baselines ----
    labels = ["no_battery", "own_battery", "central\nshared", "bilateral\nshared"]
    op_costs = [nb["op_cost"], ob["op_cost"], sh["elec_b"].sum() + sh["gas_b"].sum(), b_elec.sum() + b_gas.sum()]
    bar_colors = ["#8a8f98", "#d98a29", "#2e8b6e", "#c0392b"]
    xb = np.arange(4)
    axBar.bar(xb, op_costs, 0.6, color=bar_colors)
    for i, v in enumerate(op_costs):
        axBar.annotate(f"£{v:.2f}", (i, v), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8.5)
    axBar.set_xticks(xb)
    axBar.set_xticklabels(labels, fontsize=8.5)
    axBar.set_ylabel("community operating cost (£/day)")
    axBar.set_ylim(0, max(op_costs) * 1.18)
    axBar.set_title(
        f"Community operating cost\nbattery −£{batt_benefit:.2f}/day, sharing −£{c_flat:.2f}"
        f"/−£{b_flat:.2f} (central/bilateral)",
        fontsize=9,
        fontweight="bold",
        loc="left",
    )
    axBar.grid(True, axis="y", color="#eee", lw=0.6)

    fig.suptitle(
        f"{pd.Timestamp(SHOWCASE_DAY):%A %d %b %Y} — P2P settlement price fraction sweep, N={C.n_buildings}",
        x=0.02,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_png = PLOTS_DIR / "21_price_fraction_sweep.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"saved {out_png}")


if __name__ == "__main__":
    main(sys.argv[1:])
