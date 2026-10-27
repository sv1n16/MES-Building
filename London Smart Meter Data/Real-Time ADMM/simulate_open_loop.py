"""Real-time open loop: the day-ahead plan applied as scheduled, with no feedback.

The plan is computed once from the day-ahead forecasts (the cached bilateral
ADMM schedule) and every setpoint is then executed unchanged while the actual
load, PV and outdoor temperature of the shared noisy scenario play out:

battery   Planned charge/discharge exactly; SOC follows the plan.
heating   Planned heat-pump electrical power and boiler output. Heat-pump heat
          follows the actual COP, and the indoor temperature follows the
          optimiser's thermal model with the actual outdoor temperature.
sharing   Planned P2P trades are delivered as scheduled. A seller that is short
          covers its committed export from its own grid import.
grid      Takes the whole forecast error: each building imports whatever its
          actual balance needs; surplus is curtailed (no export to the grid).

Unlike fixed_schedule (plan plus rule-based real-time recourse), nothing reacts
to the measurements, so this is the no-feedback reference for the comparison.
"""

from __future__ import annotations

import time

import numpy as np

import simulate_realtime_reoptimise as reopt

C = reopt.C
N, T, DT = reopt.N, reopt.FULL_T, reopt.DT
METHOD = "open_loop"


def simulate_open_loop(schedule, actual_load, actual_pv, actual_t_out, *, run,
                       results_file=None, timing_file=None):
    planned = schedule["variables"]
    trades = schedule["trades"]  # trades[b, j, t] > 0: b delivers to j
    cop = C.cop_base + 0.01 * (np.asarray(actual_t_out, dtype=float) - C.T_ref)
    indoor = reopt.DAY_START_T_IN.copy()
    records, timings = [], []
    for t in range(T):
        started = time.perf_counter()
        step = []
        for b in range(N):
            charge = float(planned["charge"][b, t])
            discharge = float(planned["discharge"][b, t])
            p_hp = float(planned["p_hp"][b, t])
            q_boiler = float(planned["q_boiler"][b, t])
            if t > 0:
                indoor[b] += DT / 10 * (cop[t] * p_hp + q_boiler - 0.5 * (indoor[b] - actual_t_out[t]))
            p2p_export = float(np.maximum(trades[b, :, t], 0.0).sum())
            p2p_import = float(np.maximum(-trades[b, :, t], 0.0).sum())
            net = (float(actual_load[b, t]) + p_hp + charge + p2p_export
                   - float(actual_pv[b, t]) - discharge - p2p_import)
            final = {
                "charge": charge,
                "discharge": discharge,
                "soc": float(planned["soc"][b, t]),
                "p_el": max(net, 0.0),
                "curtail": max(-net, 0.0),
                "p_hp": p_hp,
                "q_boiler": q_boiler,
                "gas": float(planned["gas"][b, t]),
                "T_in": float(indoor[b]),
                "p2p_import": p2p_import,
                "p2p_export": p2p_export,
            }
            step.append(reopt.make_record(run, METHOD, t, b, actual_load[b, t], actual_pv[b, t],
                                          actual_t_out[t], final, schedule))
        step_timing = [reopt.timing_row(run, METHOD, t, "community", wall_seconds=time.perf_counter() - started,
                                        solver_calls=0, termination="open_loop_plan")]
        if results_file is not None:
            reopt.append_rows(results_file, step)
        if timing_file is not None:
            reopt.append_rows(timing_file, step_timing)
        records.extend(step)
        timings.extend(step_timing)
    return records, timings
