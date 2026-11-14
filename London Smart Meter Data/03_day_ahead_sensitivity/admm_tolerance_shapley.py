"""Shapley cost-saving allocation at each bilateral ADMM stopping tolerance.

A looser tolerance stops ADMM at a different iterate, so the energy shared and
the cost each coalition achieves differ by tolerance. The coalition value is
therefore evaluated with the ADMM answer itself:

    v_eps(S) = sum_{i in S} standalone cost_i  -  sum_{i in S} ADMM cost_i(S, eps)

where ADMM(S, eps) runs bilateral ADMM from a fresh state with only trades
between members of S, and stops at the first iterate meeting eq (36) at eps.
Costs are electricity + gas (P2P settlements are transfers inside S); the
energy + comfort-penalty objective is allocated as a second value.

For comparison, the same Shapley values are computed for the central sharing
optimum (``central_optimisation_showcase`` MIQP, market settlement: the
tolerance sweep's reference), solving each coalition centrally with
non-members disconnected. Its standalone costs come from the central model's
singleton solves, so each solution is measured against its own model.

The ADMM trajectory does not depend on the tolerance (it only decides when to
stop), so one run per coalition records the stopping iterate of every
tolerance. Coalition results are checkpointed and reused between runs.

The dual residual floors at ~2e-5 kW (subproblem solve accuracy), so tolerances
below that run to --max-iterations in every multi-building coalition and share
the capped iterate; lower the cap for those or the exact run takes days.

Run from this directory:

    py -3.13 admm_tolerance_shapley.py --exact --tolerances 1e-2,1e-3,1e-4
    py -3.13 admm_tolerance_shapley.py --permutations 50 --max-iterations 200
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import project_paths  # noqa: E402  (shared modules on sys.path; results -> Results/<experiment>/)

project_paths.register(__file__)

import argparse
import hashlib
import json
import math
import time

import numpy as np
import pandas as pd
import pyomo.environ as pyo


import admm_bilateral_p2p as BIL  # noqa: E402
import central_optimisation_showcase as C  # noqa: E402
import cost_shapley as CS  # noqa: E402

OUTPUT_DIR = project_paths.results_dir()  # Results/03_day_ahead_sensitivity/
DEFAULT_CSV = OUTPUT_DIR / "35_admm_tolerance_shapley_values.csv"
DEFAULT_CACHE = OUTPUT_DIR / "35_admm_tolerance_shapley_coalitions.json"
DEFAULT_TOLERANCES = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)
N, T, DT = BIL.N, BIL.T, BIL.DT
CENTRAL_KEY = "central"


def _parse_tolerances(value: str) -> list[float]:
    try:
        tolerances = [float(item.strip()) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("tolerances must be comma-separated positive numbers") from error
    if not tolerances or any(not np.isfinite(item) or item <= 0 for item in tolerances):
        raise argparse.ArgumentTypeError("tolerances must be comma-separated positive finite numbers")
    return sorted(set(tolerances), reverse=True)


def _tol_key(tolerance: float) -> str:
    return f"{tolerance:.3e}"


def _members(mask: int) -> list[int]:
    return [b for b in range(N) if mask & (1 << b)]


def _signature(max_iterations: int) -> str:
    digest = hashlib.sha256()
    for array in (C.LOAD, C.PV_B, C.PRICE, C.T_OUT, C.T_SET, C.CAP, C.MAXPOW, C.INIT_SOC, C.INIT_T_IN):
        values = np.ascontiguousarray(array)
        digest.update(str(values.shape).encode("ascii"))
        digest.update(values.tobytes())
    settings = {
        "building_ids": [str(value) for value in C.BUILDING_IDS],
        "rho": float(BIL.RHO),
        "stop_streak": int(BIL.STOP_STREAK),
        "relax_binaries": bool(BIL.RELAX_BINARIES),
        "adapt_rho": bool(BIL.ADAPT_RHO),
        "max_iterations": int(max_iterations),
        "admm_hash": hashlib.sha256(Path(BIL.__file__).read_bytes()).hexdigest(),
        "central_hash": hashlib.sha256(Path(C.__file__).read_bytes()).hexdigest(),
    }
    digest.update(json.dumps(settings, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


class CoalitionADMM:
    """Bilateral ADMM (same updates as ``BIL.run_admm``) restricted to one coalition."""

    def __init__(self) -> None:
        C._require_solver()
        self.solver = pyo.SolverFactory(C.SOLVER)
        options = [("LogFile", ""), ("OutputFlag", 0)]
        if C.SOLVE_TIME_LIMIT > 0:
            options.append(("TimeLimit", C.SOLVE_TIME_LIMIT))
        for option, value in options:
            self.solver.options[option] = value
        self.subs = [BIL.build_subproblem(b) for b in range(N)]

    def _solve(self, b: int, iteration: int) -> None:
        result = self.solver.solve(self.subs[b], tee=False)
        condition = result.solver.termination_condition
        if condition not in (pyo.TerminationCondition.optimal, pyo.TerminationCondition.locallyOptimal):
            raise RuntimeError(
                f"building {b} subproblem did not solve at ADMM iter {iteration}: "
                f"status={result.solver.status}, termination={condition}"
            )

    def _restrict(self, members: list[int]) -> None:
        """Fix trades with non-members to zero and reset duals/consensus to a fresh state."""
        inside = set(members)
        for b in members:
            m = self.subs[b]
            m.rho = float(BIL.RHO)
            for k, j in enumerate(BIL.NB[b]):
                for t in range(T):
                    m.lam[k, t] = 0.0
                    m.zloc[k, t] = 0.0
                    for variable in (m.q, m.qpos, m.qneg):
                        if j in inside:
                            variable[k, t].unfix()
                        else:
                            variable[k, t].fix(0.0)

    def _costs(self, members: list[int]) -> tuple[dict[int, float], dict[int, float]]:
        energy, objective = {}, {}
        for b in members:
            m = self.subs[b]
            p_el = np.array([pyo.value(m.p_el[t]) for t in range(T)])
            gas = np.array([pyo.value(m.gas[t]) for t in range(T)])
            t_in = np.array([pyo.value(m.T_in[t]) for t in range(T)])
            energy[b] = float((C.PRICE * p_el * DT).sum() + (C.gas_price / 100.0 * gas * DT).sum())
            objective[b] = energy[b] + float(C.alpha * ((t_in - C.T_SET) ** 2).sum())
        return energy, objective

    def run(self, mask: int, tolerances: list[float], max_iterations: int) -> dict:
        members = _members(mask)
        self._restrict(members)
        if len(members) == 1:
            self._solve(members[0], 0)
            energy, objective = self._costs(members)
            snapshot = _snapshot(energy, objective, 1, 0.0, 0.0, 0.0, True)
            return {_tol_key(tol): snapshot for tol in tolerances}

        pairs = [(lo, hi) for lo, hi in BIL.PAIRS if lo in members and hi in members]
        index = {b: {j: k for k, j in enumerate(BIL.NB[b])} for b in members}
        q_all = {b: np.zeros((N - 1, T)) for b in members}
        lam_all = {b: np.zeros((N - 1, T)) for b in members}
        Z = {p: np.zeros(T) for p in pairs}
        zloc_all = {b: np.zeros((N - 1, T)) for b in members}
        rho = float(BIL.RHO)
        npair_t = len(pairs) * T
        ncoup_t = 2 * len(pairs) * T
        streak = {tol: 0 for tol in tolerances}
        recorded: dict[str, dict] = {}

        for i in range(max_iterations):
            for b in members:
                m = self.subs[b]
                for j in members:
                    if j == b:
                        continue
                    k = index[b][j]
                    for t in range(T):
                        m.lam[k, t] = float(lam_all[b][k, t])
                        m.zloc[k, t] = float(zloc_all[b][k, t])
                self._solve(b, i)
                q_all[b] = BIL._q(m)

            Z_new = {(lo, hi): 0.5 * (q_all[lo][index[lo][hi]] - q_all[hi][index[hi][lo]]) for lo, hi in pairs}
            zloc_new = {b: np.zeros((N - 1, T)) for b in members}
            for (lo, hi), z in Z_new.items():
                zloc_new[lo][index[lo][hi]] = z
                zloc_new[hi][index[hi][lo]] = -z
            lam_new = {b: lam_all[b] + rho * (q_all[b] - zloc_new[b]) for b in members}

            prim = sum(float(np.sum((rho * (q_all[b] - zloc_new[b])) ** 2)) for b in members)
            dual = rho * sum(float(np.sum((Z_new[p] - Z[p]) ** 2)) for p in pairs)
            r_rms = float(np.sqrt(sum(float(np.sum((q_all[b] - zloc_new[b]) ** 2)) for b in members) / ncoup_t))
            s_rms = float(np.sqrt(sum(float(np.sum((Z_new[p] - Z[p]) ** 2)) for p in pairs) / npair_t))
            shared = float(sum(np.abs(Z_new[p]).sum() for p in pairs) * DT)

            pending = [tol for tol in tolerances if _tol_key(tol) not in recorded]
            energy = objective = None
            for tol in pending:
                eps1 = (rho * tol) ** 2 * ncoup_t
                eps2 = rho * tol**2 * npair_t
                streak[tol] = streak[tol] + 1 if (prim <= eps1 and dual <= eps2) else 0
                if streak[tol] >= BIL.STOP_STREAK:
                    if energy is None:
                        energy, objective = self._costs(members)
                    recorded[_tol_key(tol)] = _snapshot(energy, objective, i + 1, r_rms, s_rms, shared, True)

            Z, zloc_all, lam_all = Z_new, zloc_new, lam_new
            if len(recorded) == len(tolerances):
                break

        energy, objective = self._costs(members)
        for tol in tolerances:
            recorded.setdefault(_tol_key(tol), _snapshot(energy, objective, i + 1, r_rms, s_rms, shared, False))
        return recorded


def _snapshot(energy, objective, iterations, r_rms, s_rms, shared, converged) -> dict:
    return {
        "energy_cost": {str(b): v for b, v in energy.items()},
        "objective": {str(b): v for b, v in objective.items()},
        "iterations": int(iterations),
        "primal_rms": float(r_rms),
        "dual_rms": float(s_rms),
        "energy_shared_kwh": float(shared),
        "converged": bool(converged),
    }


def _central_run(mask: int) -> dict:
    """Central sharing optimum of one coalition (the tolerance sweep's reference model)."""
    members = _members(mask)
    model = C.build_model(True, True, fee_frac=BIL.FEE_FRAC, fee_mode="market")
    CS._exclude_nonmembers(model, set(members))
    result = C.solve_and_extract(
        model, f"central_coalition_{mask:0{N}b}", fee_frac=BIL.FEE_FRAC, fee_mode="market"
    )
    energy_b = np.asarray(result["elec_b"] + result["gas_b"], dtype=float)
    comfort_b = C.alpha * ((np.asarray(result["T_in"], dtype=float) - C.T_SET[None, :]) ** 2).sum(axis=1)
    energy = {b: float(energy_b[b]) for b in members}
    objective = {b: float(energy_b[b] + comfort_b[b]) for b in members}
    shared = float(np.asarray(result["share"], dtype=float).sum() * DT)
    return _snapshot(energy, objective, 0, 0.0, 0.0, shared, True)


class CoalitionStore:
    """Checkpointed coalition results for every ADMM tolerance and (optionally) the central optimum."""

    def __init__(self, path: Path, tolerances: list[float], max_iterations: int, central: bool) -> None:
        self.path = path
        self.tolerances = tolerances
        self.central = central
        self.max_iterations = max_iterations
        self.signature = _signature(max_iterations)
        self.results: dict[str, dict] = {}
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("schema") != 1 or payload.get("signature") != self.signature:
                raise ValueError(
                    f"Coalition cache {path} does not match the current dataset/ADMM settings. "
                    "Choose another --cache path or remove the stale cache."
                )
            self.results = payload["coalitions"]
        self.admm = CoalitionADMM()
        self._unsaved = 0

    def get(self, mask: int) -> dict:
        key = str(mask)
        cached = self.results.setdefault(key, {})
        changed = False
        if any(_tol_key(tol) not in cached for tol in self.tolerances):
            start = time.perf_counter()
            # a rerun from a fresh state reproduces cached tolerances and adds the missing ones
            cached.update(self.admm.run(mask, self.tolerances, self.max_iterations))
            iterations = max(cached[_tol_key(tol)]["iterations"] for tol in self.tolerances)
            print(
                f"  coalition {mask:0{N}b} (size {len(_members(mask))}): "
                f"ADMM {iterations} iters, {time.perf_counter() - start:.1f}s"
            )
            changed = True
        if self.central and CENTRAL_KEY not in cached:
            cached[CENTRAL_KEY] = _central_run(mask)
            changed = True
        if changed:
            self._unsaved += 1
            if self._unsaved >= 10:
                self.save()
        return cached

    def solutions(self) -> list[tuple[str, float]]:
        """(cache key, tolerance) for every solution to allocate; tolerance is NaN for central."""
        keys = [(_tol_key(tol), tol) for tol in self.tolerances]
        return keys + [(CENTRAL_KEY, float("nan"))] if self.central else keys

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema": 1, "signature": self.signature, "coalitions": self.results}
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        temporary.replace(self.path)
        self._unsaved = 0


def _baseline(store: CoalitionStore, solution: str, measure: str) -> np.ndarray:
    """Standalone (no-sharing) cost of each building, solved with the same model as ``solution``."""
    return np.array([store.get(1 << b)[solution][measure][str(b)] for b in range(N)])


def _saving(store: CoalitionStore, mask: int, solution: str, measure: str, baseline: np.ndarray) -> float:
    if mask == 0:
        return 0.0
    costs = store.get(mask)[solution][measure]
    return float(sum(baseline[b] - costs[str(b)] for b in _members(mask)))


def _shapley_exact(store, solution, measure, baseline) -> tuple[np.ndarray, np.ndarray]:
    full = (1 << N) - 1
    savings = np.array([_saving(store, mask, solution, measure, baseline) for mask in range(full + 1)])
    values = np.zeros(N)
    for b in range(N):
        bit = 1 << b
        for mask in range(full + 1):
            if mask & bit:
                continue
            size = bin(mask).count("1")
            weight = math.factorial(size) * math.factorial(N - size - 1) / math.factorial(N)
            values[b] += weight * (savings[mask | bit] - savings[mask])
    return values, np.zeros(N)


def _shapley_sampled(store, solution, measure, baseline, orders) -> tuple[np.ndarray, np.ndarray]:
    marginal = np.zeros((len(orders), N))
    for sample, order in enumerate(orders):
        mask, previous = 0, 0.0
        for b in order:
            mask |= 1 << int(b)
            saving = _saving(store, mask, solution, measure, baseline)
            marginal[sample, b] = saving - previous
            previous = saving
    errors = marginal.std(axis=0, ddof=1) / math.sqrt(len(orders)) if len(orders) > 1 else np.full(N, np.nan)
    return marginal.mean(axis=0), errors


def _solution_label(key: str, tolerance: float) -> str:
    return "central" if key == CENTRAL_KEY else f"ADMM tol {tolerance:g}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--exact", action="store_true", help="evaluate all 2^N coalitions")
    mode.add_argument("--permutations", type=int, default=50, help="sampled orderings (default: 50)")
    parser.add_argument("--seed", type=int, default=42, help="random seed for permutation sampling")
    parser.add_argument(
        "--tolerances",
        type=_parse_tolerances,
        default=list(DEFAULT_TOLERANCES),
        help="comma-separated primal/dual RMS tolerances (default: 1e-2,1e-3,1e-4,1e-5,1e-6)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=BIL.MAX_ITERS,
        help=f"ADMM iteration cap per coalition (default: {BIL.MAX_ITERS})",
    )
    parser.add_argument("--no-central", action="store_true", help="skip the central sharing optimum per coalition")
    parser.add_argument("--output", type=Path, default=DEFAULT_CSV, help="Shapley results CSV")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE, help="coalition checkpoint file")
    args = parser.parse_args()
    if not args.exact and args.permutations < 1:
        parser.error("--permutations must be at least 1")
    if args.max_iterations < 1:
        parser.error("--max-iterations must be at least 1")

    store = CoalitionStore(args.cache, args.tolerances, args.max_iterations, central=not args.no_central)
    full = (1 << N) - 1
    orders = None
    try:
        if args.exact:
            for mask in range(1, full + 1):
                store.get(mask)
        else:
            rng = np.random.default_rng(args.seed)
            orders = [rng.permutation(N) for _ in range(args.permutations)]
            for sample, order in enumerate(orders):
                mask = 0
                for b in order:
                    mask |= 1 << int(b)
                    store.get(mask)
                print(f"Completed permutation {sample + 1}/{len(orders)}")
    finally:
        store.save()

    method = "exact" if args.exact else "permutation sampling"
    rows = []
    for measure, label in (("energy_cost", "energy"), ("objective", "objective")):
        for key, tolerance in store.solutions():
            name = _solution_label(key, tolerance)
            baseline = _baseline(store, key, measure)
            if args.exact:
                values, errors = _shapley_exact(store, key, measure, baseline)
            else:
                values, errors = _shapley_sampled(store, key, measure, baseline, orders)
            total = _saving(store, full, key, measure, baseline)
            if args.exact and not np.isclose(values.sum(), total, rtol=1e-7, atol=1e-6):
                raise RuntimeError(f"Shapley values do not sum to the grand-coalition saving for {name}")
            grand = store.get(full)[key]
            for b in range(N):
                rows.append(
                    {
                        "value_measure": label,
                        "solution": name,
                        "tolerance": tolerance,
                        "grand_coalition_iterations": grand["iterations"] if key != CENTRAL_KEY else np.nan,
                        "grand_coalition_converged": grand["converged"],
                        "grand_coalition_energy_shared_kwh_per_day": grand["energy_shared_kwh"],
                        "building_id": C.BUILDING_IDS[b],
                        "assets": C.ASSETS[b],
                        "standalone_cost_gbp_per_day": baseline[b],
                        "grand_coalition_cost_gbp_per_day": grand[measure][str(b)],
                        "shapley_saving_gbp_per_day": values[b],
                        "shapley_standard_error_gbp_per_day": errors[b],
                        "shapley_allocated_cost_gbp_per_day": baseline[b] - values[b],
                        "share_of_total_saving_pct": 100.0 * values[b] / total if abs(total) > 1e-9 else np.nan,
                        "total_community_saving_gbp_per_day": total,
                        "estimation": method,
                    }
                )

    table = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)

    coalitions = []
    for mask_key, by_solution in store.results.items():
        mask = int(mask_key)
        for key, tolerance in store.solutions():
            entry = by_solution.get(key)
            if entry is None:  # cached from an earlier run with other settings, not revisited
                continue
            admm = key != CENTRAL_KEY
            coalitions.append(
                {
                    "coalition_mask": mask,
                    "coalition_size": len(_members(mask)),
                    "members": ";".join(C.BUILDING_IDS[b] for b in _members(mask)),
                    "solution": _solution_label(key, tolerance),
                    "tolerance": tolerance,
                    "iterations": entry["iterations"] if admm else np.nan,
                    "converged": entry["converged"],
                    "final_primal_rms_kw": entry["primal_rms"] if admm else np.nan,
                    "final_dual_rms_kw": entry["dual_rms"] if admm else np.nan,
                    "energy_shared_kwh_per_day": entry["energy_shared_kwh"],
                    "energy_cost_gbp_per_day": sum(entry["energy_cost"].values()),
                    "objective_gbp_per_day": sum(entry["objective"].values()),
                }
            )
    coalition_csv = args.output.with_name(f"{args.output.stem}_coalitions.csv")
    pd.DataFrame(coalitions).sort_values(["coalition_mask", "solution"]).to_csv(coalition_csv, index=False)

    print(f"\nSaved {args.output}\nSaved {coalition_csv}")
    pivot = table[table["value_measure"] == "energy"].pivot(
        index="building_id", columns="solution", values="shapley_saving_gbp_per_day"
    )
    print("\nEnergy-cost Shapley savings (GBP/day) by solution:")
    print(pivot.round(4).to_string())


if __name__ == "__main__":
    main()
