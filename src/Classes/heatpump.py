import numpy as np
import pandas as pd
import pyomo.environ as pyo


class HeatPump:

    def __init__(self, time_horizon, cop=None, p_th_nom=12, lower_activation_limit=1):
        q_nominal = np.zeros(time_horizon)

        if cop is None:
            cop = np.ones(time_horizon) * 2.18
        self.cop = cop

        timesteps_total = time_horizon
        timesteps_used_horizon = time_horizon
        self.total_p_consumption = np.zeros(timesteps_total)
        self.current_p_consumption = np.zeros(timesteps_used_horizon)
        self.p_th_heat = np.zeros(timesteps_used_horizon)
        self.p_el_heat = np.zeros(timesteps_used_horizon)
        self.p_el_schedule = np.zeros(timesteps_used_horizon)
        self.p_th_nom = p_th_nom

    def set_parameters(self, model):
        model.cop = pyo.Param(model.t, mutable=True)
        for t in model.t:
            model.cop[t] = float(self.cop[t])
        model.p_th_heat_vars = pyo.Var(model.t, bounds=(-self.p_th_nom, 0), initialize=0)  # Thermal power of heat pump
        model.p_el_vars = pyo.Var(
            model.t, domain=pyo.Reals, bounds=(0, None), initialize=0
        )  # Electrical power of heat pump
        model.hp_on = pyo.Var(model.t, within=pyo.Binary)

    def set_constraints(self, model):

        def p_coupl_rule(model, t):
            return model.p_th_heat_vars[t] + model.cop[t] * model.p_el_vars[t] == 0

        model.p_coupl_constr = pyo.Constraint(model.t, rule=p_coupl_rule)

        def hp_output_onoff_rule(model, t):
            return model.p_th_heat_vars[t] >= -self.p_th_nom * model.hp_on[t]

        model.hp_output_onoff_min = pyo.Constraint(model.t, rule=hp_output_onoff_rule)
