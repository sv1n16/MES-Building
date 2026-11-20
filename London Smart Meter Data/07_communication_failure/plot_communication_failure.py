"""How robust is rolling ADMM to communication failures, compared with rolling central?

PURPOSE
    Figures only: the extra cost caused by communication failures, per method.
    impact(method, level, run) = cost(method, level, run) − cost(method, level 0, run)
    cost = electricity + gas only (no comfort penalty)

Reads the saved results of simulate_communication_failure.py and shows only the impact
of the failures: a method's cost with failures minus the same method's cost on the same
noise scenario without failures (level 0). Three questions, three folders:

    Results/07_communication_failure/1_no_fallback/        (building and link failures)
        Rolling ADMM against rolling central with no fallback: a building the
        coordinator cannot reach gets no set-points and runs uncontrolled (battery idle,
        no trades, basic thermostat).
    Results/07_communication_failure/2_with_fallback/      (building and link failures)
        Rolling ADMM against rolling central with a fallback: the cut-off building
        follows its last central plan, or re-optimises itself locally.
    Results/07_communication_failure/3_central_entity_failure/   (coordinator test)
        The central coordinator is down for h hours: every building is cut off from it.
        Rolling ADMM has no coordinator and is unaffected; rolling central without a
        fallback leaves every building uncontrolled, with a fallback every building
        follows its last plan or re-optimises locally.

In each folder, per test (<test>/):
    90_impact_summary.png                      daily impact per method and failure level, and
                                               central's impact minus ADMM's (bars, mean ± std
                                               over scenarios)
    91_impact_timeseries_level_<x>_run_<n>.png one per failure level and scenario: when each
                                               building is offline, the hourly impact and the
                                               cumulative impact over the day
and across tests:
    92_impact_overview.png (.csv)              daily impact at each test's highest level

Only (level, scenario) pairs that every method of the folder has finished are used, so
the methods are always compared on the same scenarios. It can run while the suite is
still going.

Examples:
    python plot_communication_failure.py
    python plot_communication_failure.py --test building link
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import project_paths  # noqa: E402  (shared modules on sys.path; results -> Results/<experiment>/)

project_paths.register(__file__)

import argparse  # noqa: E402
import textwrap  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

import central_optimisation_showcase as C  # noqa: E402
import compare_realtime_methods as compare  # noqa: E402
from plot_realtime_comparison import INK, PLAN_COLOR, _style  # noqa: E402
from plot_realtime_comparison import _save as _save_with_retries  # noqa: E402
from simulate_communication_failure import (  # noqa: E402
    ADMM_FALLBACK, CENTRAL_LOCAL, CENTRAL_NONE, COLORS, LABELS, LEVEL_LABELS, LEVEL_SYMBOLS, LINESTYLES, MARKERS,
    HATCHES, METHOD_TITLES, RESULTS_ROOT, TESTS, Outputs,
)

N, T = C.n_buildings, C.time_horizon


def _save(fig, path):
    """Save a figure; if the file stays locked (open preview, OneDrive sync), save it as <name>_new and warn."""
    try:
        _save_with_retries(fig, path)
    except OSError as error:
        fallback = Path(path).with_name(Path(path).stem + "_new" + Path(path).suffix)
        _save_with_retries(fig, fallback)
        print(f"  warning: could not write {Path(path).name} ({error.strerror}); saved {fallback.name} instead. "
              f"Close any preview of it and re-run to replace it.")
ADMM = "rolling_admm"
BUILDING_AND_LINK_TESTS = tuple(name for name, test in TESTS.items() if test["mode"] != "coordinator")
FOLDERS = {
    "1_no_fallback": {
        "methods": (ADMM, ADMM_FALLBACK, CENTRAL_NONE),
        "tests": BUILDING_AND_LINK_TESTS,
        "question": "rolling ADMM (without and with the iteration-budget fallback) vs rolling central with no fallback",
    },
    "2_with_fallback": {
        "methods": (ADMM, ADMM_FALLBACK, "rolling_central", CENTRAL_LOCAL),
        "tests": BUILDING_AND_LINK_TESTS,
        "question": "rolling ADMM vs rolling central, each without and with a fallback",
    },
    "3_central_entity_failure": {
        "methods": (ADMM, ADMM_FALLBACK, CENTRAL_NONE, "rolling_central", CENTRAL_LOCAL),
        "tests": ("coordinator",),
        "question": "the central coordinator fails: rolling ADMM (no coordinator) vs rolling central "
                    "without and with a fallback",
    },
}


# ============================================================================
# DATA
# ============================================================================
def load_impact(name, methods):
    """Per (level, run, method, interval, building): cost with failures minus without.

    Only complete runs, and only (level, run) pairs that every method has finished.
    """
    out = Outputs(RESULTS_ROOT / name)
    if not (out.results.exists() and out.results.stat().st_size):
        return None
    data = pd.read_csv(out.results)
    data = data[data["method"].isin(methods)]
    keys = ["failure_level", "simulation_run", "method"]
    data = data[data.groupby(keys)["interval"].transform("size") == T * N]
    finished = data.groupby(["failure_level", "simulation_run"])["method"].transform("nunique") == len(methods)
    data = data[finished].copy()
    if data.empty or not (data["failure_level"] == 0).any() or not (data["failure_level"] != 0).any():
        return None
    # energy bill only: electricity + gas, no comfort penalty (the P2P settlement cancels in the community)
    data["energy_cost"] = data["final_electricity_cost"] + data["final_gas_cost"]
    pair = ["simulation_run", "method", "interval", "building"]
    reference = data[data["failure_level"] == 0].set_index(pair)["energy_cost"].rename("reference_cost")
    data = data.join(reference, on=pair).dropna(subset=["reference_cost"])
    data["impact"] = data["energy_cost"] - data["reference_cost"]
    data["building"] = data["building"].astype(str)
    return data[data["failure_level"] != 0]


def daily_impact(data):
    """Community impact per day: (failure_level, simulation_run, method) -> GBP."""
    return data.groupby(["failure_level", "simulation_run", "method"])["impact"].sum().rename("impact").reset_index()


# references for the difference figures: rolling ADMM, and ADMM + fallback where it was run
GAP_REFERENCES = {ADMM: "_minus_rolling_admm", ADMM_FALLBACK: "_minus_admm_fallback"}


def with_central_minus_admm(daily, methods):
    """Add pseudo-methods "gap:<reference>:<method>": a method's daily impact minus the reference's
    (same level and run), for every reference in GAP_REFERENCES that the folder has."""
    wide = daily.pivot_table(index=["failure_level", "simulation_run"], columns="method", values="impact")
    extra = []
    for reference in GAP_REFERENCES:
        if reference not in methods or reference not in wide:
            continue
        for method in methods:  # the reference too (a zero bar), so every method stays in every figure
            if method not in wide:
                continue
            gap = (wide[method] - wide[reference]).rename("impact").reset_index()
            extra.append(gap.assign(method=f"gap:{reference}:{method}"))
    return pd.concat([daily, *extra], ignore_index=True) if extra else daily


def _level_text(test, level):
    return f"{LEVEL_SYMBOLS[test['mode']]} = {level:g}"


# what the failure-level symbol on the axes means, per test mode
LEVEL_KEYS = {
    "building": "k = buildings offline every hour (cut off from their peers for ADMM, from the coordinator for central)",
    "link": "p = probability that each peer-to-peer link fails in an hour",
    "coordinator": "h = consecutive hours the central coordinator is down (ADMM has no coordinator)",
}


def _level_key(modes):
    """Key line(s) for the level symbols used in a figure, one per test mode."""
    return "\n".join(LEVEL_KEYS[mode] for mode in dict.fromkeys(modes))


# ============================================================================
# FIGURES
# ============================================================================
def _grouped_bars(axis, table, groups, keys, *, group_column="failure_level", colour=None, label=None,
                  centres=False):
    """Mean ± std over scenarios of the daily impact: one cluster per group, one bar per key (method)."""
    colour = colour or (lambda key: COLORS[key])
    label = label or (lambda key: METHOD_TITLES[key.split(":")[-1]])
    _style(axis)
    x = np.asarray(groups, dtype=float) if centres else np.arange(len(groups))
    width = 0.8 / len(keys)
    annotate = len(groups) * len(keys) <= 30
    for i, key in enumerate(keys):
        method = key.split(":")[-1]
        stats = (table[table["method"] == key].groupby(group_column)["impact"]
                 .agg(["mean", "std"]).reindex(groups))
        spread = stats["std"].fillna(0.0)
        positions = x + (i - (len(keys) - 1) / 2) * width
        axis.bar(positions, stats["mean"], width * 0.92, yerr=spread, capsize=2, color=colour(key),
                 hatch=HATCHES.get(method), edgecolor="white", linewidth=0, label=label(key),
                 error_kw={"elinewidth": 1, "ecolor": PLAN_COLOR})
        if not annotate:
            continue
        for position, value, sd in zip(positions, stats["mean"], spread):
            if np.isfinite(value):  # label beyond the end of the error bar
                end = value + sd if value >= 0 else value - sd
                axis.annotate(f"{value:+.2f}", (position, end), xytext=(0, 3 if value >= 0 else -3),
                              textcoords="offset points", ha="center", va="bottom" if value >= 0 else "top",
                              fontsize=7, color=INK)
    axis.axhline(0, color=PLAN_COLOR, lw=1)
    axis.set_xticks(x)
    axis.grid(False, axis="x")
    axis.margins(y=0.15)


def _impact_figures(table, groups, tick_labels, methods, *, group_column, xlabel, title, path):
    """Figures with grouped bars on one axis, no strip of panels: each method's daily impact (path),
    and for every reference in GAP_REFERENCES each other method's impact minus the reference's
    (path with the reference's suffix, e.g. _minus_rolling_admm)."""
    present = set(table["method"])
    width = min(max(4.0 + len(groups) * (0.35 * len(methods) + 0.4), 6.5), 13.0)
    rotate = len(groups) > 4 or max(len(t) for t in tick_labels) > 14
    written = []
    figures = [(list(methods), "", "Daily impact of the failures (GBP)\nelectricity + gas, with failures − without",
                None, "")]
    for reference, suffix in GAP_REFERENCES.items():
        name = {ADMM: "rolling ADMM", ADMM_FALLBACK: "ADMM + fallback"}[reference]
        figures.append((
            [f"gap:{reference}:{m}" for m in methods if f"gap:{reference}:{m}" in present], suffix,
            f"Impact − {name}'s impact (GBP/day)\n+ = {name} more robust",
            lambda key: METHOD_TITLES[key.split(":")[-1]],  # the reference is named in the title and y axis
            f"\neach method minus {METHOD_TITLES[reference]}",
        ))
    for keys, suffix, ylabel, legend, subtitle in figures:
        if not keys:
            continue
        name_of = legend or (lambda key: METHOD_TITLES[key])
        if len(groups) == 1:
            # one test or level: one bar per method, named on the axis, no legend needed
            fig, axis = plt.subplots(figsize=(max(1.6 * len(keys) + 2.0, 5.5), 5.2))
            single = table[table[group_column] == groups[0]]
            for i, key in enumerate(keys):
                _grouped_bars(axis, single.assign(_slot=i), [i], [key], group_column="_slot",
                              colour=lambda k: COLORS[k.split(":")[-1]], centres=True)
            axis.set_xticks(range(len(keys)))
            axis.set_xticklabels([textwrap.fill(name_of(k), 16) for k in keys], fontsize=8)
            axis.set_xlim(-0.6, len(keys) - 0.4)
            axis.set_xlabel(tick_labels[0].replace("\n", " ") + "\n" + xlabel, fontsize=9, color=INK)
        else:
            fig, axis = plt.subplots(figsize=(width, 5.6))
            _grouped_bars(axis, table, groups, keys, group_column=group_column,
                          colour=lambda key: COLORS[key.split(":")[-1]], label=legend)
            axis.set_xticklabels(tick_labels, fontsize=8, rotation=30 if rotate else 0,
                                 ha="right" if rotate else "center")
            axis.set_xlabel(xlabel, fontsize=9, color=INK)
            fig.legend(*axis.get_legend_handles_labels(), frameon=False, fontsize=8, loc="lower center",
                       ncol=min(len(keys), 4 if len(keys) <= 4 else 2))
        axis.set_ylabel(ylabel, fontsize=9, color=INK)
        full = title + subtitle
        axis.set_title("\n".join(textwrap.fill(line, 95) for line in full.split("\n")), fontsize=10, color=INK)
        fig.tight_layout(rect=(0, 0 if len(groups) == 1 else 0.07, 1, 1))
        target = path.with_name(path.stem + suffix + path.suffix)
        target.parent.mkdir(parents=True, exist_ok=True)
        _save(fig, target)
        plt.close(fig)
        written.append(target)
    return written


def plot_summary(data, test, methods, folder, name):
    daily = with_central_minus_admm(daily_impact(data), methods)
    levels = sorted(daily["failure_level"].unique())
    runs = data["simulation_run"].nunique()
    title = (f"{name.replace('_', ' ')}: {test['description']}\n"
             f"N = {N}, mean ± std over {runs} scenario{'s' if runs != 1 else ''}")
    paths = _impact_figures(daily, levels, [_level_text(test, level) for level in levels], methods,
                            group_column="failure_level", xlabel=_level_key([test["mode"]]), title=title,
                            path=RESULTS_ROOT / folder / name / "90_impact_summary.png")
    return paths[0] if paths else None


def _building_label(building):
    b = [str(x) for x in C.BUILDING_IDS].index(str(building))
    assets = [name for name, has in (("battery", C.CAP[b] > 1e-9), ("PV", C.PV_B[b].sum() > 1e-9)) if has]
    return f"{building} ({' + '.join(assets) if assets else 'no battery/PV'})"


def plot_timeseries(data, test, methods, folder, name, level, run):
    """When each building is offline, and what it costs each method hour by hour and over the day."""
    at = data[(data["failure_level"] == level) & (data["simulation_run"] == run)]
    if at.empty:
        return None
    hours = np.arange(T)
    buildings = [str(b) for b in C.BUILDING_IDS]
    fig, (map_axis, cost_axis, total_axis) = plt.subplots(
        3, 1, figsize=(12, 10), sharex=True, gridspec_kw={"height_ratios": [1.5, 1.3, 1.3]})

    # a) failures: peer-to-peer links lost (what ADMM sees) shaded, cut off from the coordinator crossed
    admm = at[at["method"] == ADMM]
    share = (admm.pivot_table(index="building", columns="interval", values="lost_links", aggfunc="sum")
             .reindex(index=buildings, columns=hours).fillna(0.0) / max(N - 1, 1))
    map_axis.imshow(share.to_numpy(), cmap=LinearSegmentedColormap.from_list("lost", ["#f0efec", "#1baf7a"]),
                    vmin=0, vmax=1, aspect="auto", interpolation="nearest",
                    extent=(-0.5, T - 0.5, N - 0.5, -0.5))
    central = at[at["method"] == next(m for m in methods if m != ADMM)]
    cut = central[central["building_offline"].astype(bool)]
    map_axis.scatter(cut["interval"], [buildings.index(b) for b in cut["building"]], marker="x", s=40,
                     color="#8b78b5", linewidths=2)
    map_axis.set_yticks(range(N))
    map_axis.set_yticklabels([_building_label(b) for b in buildings], fontsize=7, color=INK)
    map_axis.set_xticks(np.arange(-0.5, T, 1), minor=True)
    map_axis.set_yticks(np.arange(-0.5, N, 1), minor=True)
    map_axis.grid(which="minor", color="white", lw=0.8)
    map_axis.grid(which="major", visible=False)
    map_axis.tick_params(which="both", length=0)
    for side in map_axis.spines.values():
        side.set_visible(False)
    map_axis.set_title("a) When each building is offline. Green: share of its peer-to-peer links lost "
                       "(ADMM). Purple x: cut off from the central coordinator", loc="left", fontsize=10, color=INK)

    # b) hourly impact, c) cumulative impact
    hourly = at.groupby(["method", "interval"])["impact"].sum()
    for axis in (cost_axis, total_axis):
        _style(axis)
        axis.axhline(0, color=PLAN_COLOR, lw=1)
    for method in methods:
        series = hourly.loc[method].reindex(hours, fill_value=0.0)
        style = {"color": COLORS[method], "lw": 2, "marker": MARKERS[method], "ms": 4,
                 "ls": LINESTYLES.get(method, "-"), "label": f"{LABELS[method]}: {series.sum():+.2f} GBP/day"}
        cost_axis.plot(hours, series, **style)
        total_axis.plot(hours, series.cumsum(), **style)
    cost_axis.set_ylabel("GBP per hour", fontsize=9, color=INK)
    cost_axis.set_title("b) Impact of the failures hour by hour (community electricity + gas cost with failures − without)",
                        loc="left", fontsize=10, color=INK)
    total_axis.set_ylabel("GBP since midnight", fontsize=9, color=INK)
    total_axis.set_title("c) Cumulative impact over the day", loc="left", fontsize=10, color=INK)
    total_axis.set_xlabel("Hour of day", fontsize=9, color=INK)
    total_axis.set_xticks(hours[::2])

    fig.suptitle(f"{name.replace('_', ' ')}: {test['description']} ({_level_text(test, level)}), scenario {run}"
                 f" (N = {N})", fontsize=12, color=INK)
    handles, labels = total_axis.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=1, frameon=False, fontsize=8,
               title="Impact over the day", title_fontsize=8)
    fig.tight_layout(rect=(0, 0.04 + 0.022 * len(methods), 1, 0.96))
    path = RESULTS_ROOT / folder / name / f"91_impact_timeseries_level_{level:g}_run_{run}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    _save(fig, path)
    plt.close(fig)
    return path


def plot_overview(bundles, methods, folder):
    """Daily impact at each test's highest level, one panel per method plus central − ADMM."""
    rows = []
    for name, data in bundles.items():
        test = TESTS[name]
        level = data["failure_level"].max()
        daily = with_central_minus_admm(daily_impact(data[data["failure_level"] == level]), methods)
        rows.append(daily.assign(test=name, label=f"{name.replace('_', ' ')}\n({_level_text(test, level)})"))
    table = pd.concat(rows, ignore_index=True)
    summary = table.groupby(["test", "failure_level", "method"])["impact"].agg(["mean", "std", "count"])
    summary.reset_index().to_csv(RESULTS_ROOT / folder / "92_impact_overview.csv", index=False)

    labels = list(dict.fromkeys(table["label"]))
    paths = _impact_figures(
        table, labels, labels, methods, group_column="label", xlabel="Test (at its highest failure level)\n" + _level_key(TESTS[n]["mode"] for n in bundles),
        title=f"{FOLDERS[folder]['question']}\nhighest failure level of each test, N = {N}, mean ± std over scenarios",
        path=RESULTS_ROOT / folder / "92_impact_overview.png")
    return paths[0] if paths else None


# ============================================================================
# COMPUTATION TIME (4_computation_time/)
# ============================================================================
TIMING_FOLDER = "4_computation_time"
# every central variant is timed separately: the coordinator solve is the same, but the
# last-plan fallback fixes the cut-off buildings' set-points and the local fallback adds
# the cut-off buildings' own solves to the hour
TIMING_METHODS = (ADMM, ADMM_FALLBACK, CENTRAL_NONE, "rolling_central", CENTRAL_LOCAL)
ADMM_METHODS = (ADMM, ADMM_FALLBACK)  # timed as the parallel estimate (one solver per building)
LIMITS = ((60.0, "1 min"), (3600.0, "1 h (the interval)"))


def _seconds(frame, method):
    """Per-hour time: ADMM's parallel estimate (one solver per building), central's wall time."""
    return frame["parallel_latency_seconds" if method in ADMM_METHODS else "wall_seconds"]


def load_timings(name):
    """Per-hour step times of ADMM and every central variant (hours 1-23; hour 0 reproduces the plan).

    Only (level, run) pairs that every method has finished.
    """
    out = Outputs(RESULTS_ROOT / name)
    if not (out.timings.exists() and out.timings.stat().st_size):
        return None
    t = pd.read_csv(out.timings)
    t = t[t["method"].isin(TIMING_METHODS)]
    # remove machine sleep/lock pauses (Windows keeps the clock running): ADMM iterations more than
    # 20x their interval's median are replaced by it, as in experiment 05, level by level
    t = pd.concat([compare.clean_suspended_timings(group.to_dict("records")).assign(failure_level=level)
                   for level, group in t.groupby("failure_level")], ignore_index=True)
    t = t[(t["scope"] == "community") & (t["interval"] > 0)].copy()
    # any other step still caught by a pause (e.g. a central solve): drop it and say so
    median = t.groupby(["failure_level", "method"])["parallel_latency_seconds"].transform("median")
    paused = t["parallel_latency_seconds"] > compare.SUSPEND_FACTOR * median.clip(lower=0.05)
    if paused.any() or t["suspended"].any():
        print(f"  {name}: {int(t['suspended'].sum())} step(s) corrected for a machine pause, "
              f"{int(paused.sum())} dropped")
    t = t[~paused]
    t = t[t.groupby(["failure_level", "simulation_run", "method"])["interval"].transform("size") == T - 1]
    present = t["method"].nunique()
    every = t.groupby(["failure_level", "simulation_run"])["method"].transform("nunique") == present
    t = t[every]
    t["admm_iterations"] = pd.to_numeric(t["admm_iterations"], errors="coerce")
    return t if not t.empty else None


def _boxes(axis, groups, positions, colour, hatch=None, width=0.35):
    plot = axis.boxplot([g.dropna().to_numpy() for g in groups], positions=positions, widths=width,
                        patch_artist=True, showfliers=True, medianprops={"color": INK, "lw": 1.2},
                        flierprops={"marker": "o", "markersize": 2.5, "markerfacecolor": colour,
                                    "markeredgecolor": colour, "alpha": 0.6},
                        whiskerprops={"color": PLAN_COLOR}, capprops={"color": PLAN_COLOR})
    for patch in plot["boxes"]:
        patch.set_facecolor(colour)
        patch.set_edgecolor("white")
        if hatch:
            patch.set_hatch(hatch)
    return plot["boxes"][0] if plot["boxes"] else None


def _limit_lines(axis):
    for seconds, label in LIMITS:
        axis.axhline(seconds, color=PLAN_COLOR, lw=1, ls=":")
        axis.annotate(label, (1.0, seconds), xycoords=("axes fraction", "data"), xytext=(-2, 2),
                      textcoords="offset points", ha="right", va="bottom", fontsize=7, color=PLAN_COLOR)


TIMING_TITLES = {ADMM: "Rolling ADMM (parallel estimate)",
                 ADMM_FALLBACK: "ADMM + iteration-budget fallback (incl. the local solves)", CENTRAL_NONE: "Central, no fallback",
                 "rolling_central": "Central + last-plan fallback",
                 CENTRAL_LOCAL: "Central + local fallback (incl. the local solves)"}


def _method_boxes(axis, t, groups, group_column, value_of, methods):
    """Grouped boxes on one axis: one cluster per group, one box per method. Returns legend handles."""
    x = np.arange(len(groups))
    width = 0.8 / len(methods)
    handles = []
    methods = [m for m in methods if (t["method"] == m).any()]
    width = 0.8 / max(len(methods), 1)
    for i, method in enumerate(methods):
        part = t[t["method"] == method]
        values = [value_of(part[part[group_column] == g], method) for g in groups]
        box = _boxes(axis, values, x + (i - (len(methods) - 1) / 2) * width, COLORS[method],
                     hatch=HATCHES.get(method), width=width * 0.85)
        handles.append((box, TIMING_TITLES[method]))
    axis.set_xticks(x)
    axis.grid(False, axis="x")
    return [h for h in handles if h[0] is not None]


def _finish(fig, axis, handles, title, path):
    axis.set_title(title, fontsize=10, color=INK)
    fig.legend([h for h, _ in handles], [label for _, label in handles], frameon=False, fontsize=8,
               loc="lower center", ncol=min(len(handles), 4 if fig.get_figwidth() >= 11 else 2))
    fig.tight_layout(rect=(0, 0.07 if fig.get_figwidth() >= 11 or len(handles) <= 2 else 0.11, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    _save(fig, path)
    plt.close(fig)
    return path


def plot_timing(name, t):
    """Two figures per test: per-hour computation time of every method, and ADMM's iterations."""
    test = TESTS[name]
    levels = sorted(t["failure_level"].unique())
    labels = [_level_text(test, level) for level in levels]
    runs = t["simulation_run"].nunique()
    scope = (f"{name.replace('_', ' ')}: {test['description']}\nN = {N}, every hour 1-23 of {runs} "
             f"scenario{'s' if runs != 1 else ''}; boxes: median and quartiles, whiskers 1.5 IQR")
    width = min(max(3.0 + 1.4 * len(levels), 7.0), 12.0)

    fig, axis = plt.subplots(figsize=(width, 5.6))
    _style(axis)
    axis.set_yscale("log")
    _limit_lines(axis)
    handles = _method_boxes(axis, t, levels, "failure_level", lambda frame, m: _seconds(frame, m), TIMING_METHODS)
    axis.set_xticklabels(labels, fontsize=8)
    axis.set_xlabel(_level_key([test["mode"]]), fontsize=9, color=INK)
    axis.set_ylabel("Computation time per real-time hour (s, log scale)", fontsize=9, color=INK)
    axis.set_ylim(top=3600 * 5)
    path = _finish(fig, axis, handles, f"Computation time per hour\n{scope}",
                   RESULTS_ROOT / TIMING_FOLDER / name / "93_computation_time.png")

    fig, axis = plt.subplots(figsize=(width, 5.2))
    _style(axis)
    handles = _method_boxes(axis, t, levels, "failure_level", lambda frame, m: frame["admm_iterations"], ADMM_METHODS)
    axis.set_xticklabels(labels, fontsize=8)
    axis.set_xlabel(_level_key([test["mode"]]), fontsize=9, color=INK)
    axis.set_ylabel("ADMM iterations per hour", fontsize=9, color=INK)
    _finish(fig, axis, handles, f"ADMM iterations per hour (the fallback variant stops at its budget)\n{scope}",
            RESULTS_ROOT / TIMING_FOLDER / name / "93_admm_iterations.png")
    return path


def timing_table(tables):
    """Per test and level: ADMM iterations and parallel time, each central variant's time (s per hour)."""
    rows = []
    for name, t in tables.items():
        for level, g in t.groupby("failure_level"):
            admm = g[g["method"] == ADMM]
            row = {"test": name, "failure_level": level, "scenarios": g["simulation_run"].nunique(),
                   "admm_iterations_median": admm["admm_iterations"].median(),
                   "admm_iterations_max": admm["admm_iterations"].max()}
            for method in TIMING_METHODS:
                seconds = _seconds(g[g["method"] == method], method)
                row[f"{method}_s_median"] = seconds.median()
                row[f"{method}_s_max"] = seconds.max()
            rows.append(row)
    path = RESULTS_ROOT / TIMING_FOLDER / "95_computation_time_table.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def plot_timing_overview(tables):
    """Per-hour time of every method at each test's highest failure level, on one axis."""
    rows, frames = [], []
    for name, t in tables.items():
        highest = t["failure_level"].max()
        test = TESTS[name]
        label = f"{name.replace('_', ' ')}\n({_level_text(test, highest)})"
        for level in (0, highest):
            part = t[t["failure_level"] == level]
            for method in TIMING_METHODS:
                values = _seconds(part[part["method"] == method], method)
                rows.append({"test": name, "level": level, "method": method, "median_s": values.median(),
                             "p95_s": values.quantile(0.95), "max_s": values.max(), "hours": len(values)})
        frames.append(t[t["failure_level"] == highest].assign(label=label))
    pd.DataFrame(rows).to_csv(RESULTS_ROOT / TIMING_FOLDER / "94_computation_time_overview.csv", index=False)

    table = pd.concat(frames, ignore_index=True)
    labels = list(dict.fromkeys(table["label"]))
    fig, axis = plt.subplots(figsize=(min(max(3.0 + 1.6 * len(labels), 7.0), 13.0), 5.8))
    _style(axis)
    axis.set_yscale("log")
    _limit_lines(axis)
    handles = _method_boxes(axis, table, labels, "label", lambda frame, m: _seconds(frame, m), TIMING_METHODS)
    axis.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
    axis.set_xlabel("Test (at its highest failure level)\n" + _level_key(TESTS[n]["mode"] for n in tables),
                   fontsize=9, color=INK)
    axis.set_ylabel("Computation time per real-time hour (s, log scale)", fontsize=9, color=INK)
    axis.set_ylim(top=3600 * 5)
    return _finish(fig, axis, handles,
                   f"Computation time per hour at each test's highest failure level (N = {N})\n"
                   f"rolling ADMM vs rolling central, each without and with a fallback",
                   RESULTS_ROOT / TIMING_FOLDER / "94_computation_time_overview.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test", nargs="+", choices=tuple(TESTS), default=list(TESTS))
    parser.add_argument("--folder", nargs="+", choices=(*FOLDERS, TIMING_FOLDER),
                        default=[*FOLDERS, TIMING_FOLDER])
    args = parser.parse_args()

    if TIMING_FOLDER in args.folder:
        timing_tables = {}
        for name in args.test:
            t = load_timings(name)
            if t is None or t["failure_level"].nunique() < 2:
                continue
            timing_tables[name] = t
            print(f"{TIMING_FOLDER}: {plot_timing(name, t)}")
        if timing_tables:
            print(f"{TIMING_FOLDER}: {plot_timing_overview(timing_tables)}")
            print(f"{TIMING_FOLDER}: {timing_table(timing_tables)}")

    for folder in [f for f in args.folder if f in FOLDERS]:
        methods = FOLDERS[folder]["methods"]
        print(f"{folder}: {FOLDERS[folder]['question']}")
        bundles = {}
        for name in [t for t in args.test if t in FOLDERS[folder]["tests"]]:
            data = load_impact(name, methods)
            if data is None:
                print(f"  {name}: no failure level finished by every method yet; skipped.")
                continue
            bundles[name] = data
            test = TESTS[name]
            paths = [plot_summary(data, test, methods, folder, name)]
            for level in sorted(data["failure_level"].unique()):
                for run in sorted(data.loc[data["failure_level"] == level, "simulation_run"].unique()):
                    paths.append(plot_timeseries(data, test, methods, folder, name, level, run))
            print(f"  {name}: {len([p for p in paths if p])} figures in {RESULTS_ROOT / folder / name}")
        if bundles:
            print(f"  overview: {plot_overview(bundles, methods, folder)}")


if __name__ == "__main__":
    main()
