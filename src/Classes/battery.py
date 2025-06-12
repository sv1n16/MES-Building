import numpy as np
import pandas as pd
import pyomo.environ as pyo


class Battery:

    def __init__(
        self,
        max_discharge_power=4.6,
        max_charge_power=4.6,
        initial_soc=0.8,
        max_capacity=12.0,
        eta_charge=0.9,  # Charging efficiency
        eta_discharge=0.9,
        p_el_demand=None,
        p_el_supply=None,
    ):
        self.max_charge_power = max_charge_power
        self.max_discharge_power = max_discharge_power
        self.initial_soc = max_capacity * initial_soc
        self.max_capacity = max_capacity
        self.eta = eta_charge

        self.p_el_charge_schedule = p_el_demand
        self.p_el_discharge_schedule = p_el_supply
        self.energy_el_schedule = None  # state of charge
        self.power_el_schedule = None  # charge and discharge schedule

    def set_parameters(self, model):
        model.charge = pyo.Var(model.t, bounds=(0, self.max_charge_power), initialize=0)  # Charging power
        model.discharge = pyo.Var(model.t, bounds=(0, self.max_discharge_power), initialize=0)  # Discharging power
        model.soc = pyo.Var(model.t, bounds=(0, self.max_capacity), initialize=self.initial_soc)  # State of charge
        model.charging_state = pyo.Var(model.t, domain=pyo.Binary)

    def set_constraints(self, model):
        # Constraint for charging power
        def c_rule(model, t):
            return model.charge[t] <= model.charging_state[t] * self.max_charge_power

        model.max_charge_constr = pyo.Constraint(model.t, rule=c_rule)

        # Constraint for discharging power
        def d_rule(model, t):
            return model.discharge[t] <= (1 - model.charging_state[t]) * self.max_discharge_power

        model.max_discharge_constr = pyo.Constraint(model.t, rule=d_rule)

        # Ensure charge and discharge start at 0
        def no_charge_at_start_rule(model):
            return model.charge[0] == 0

        model.no_charge_at_start = pyo.Constraint(rule=no_charge_at_start_rule)

        def no_discharge_at_start_rule(model):
            return model.discharge[0] == 0

        model.no_discharge_at_start = pyo.Constraint(rule=no_discharge_at_start_rule)

        def soc_rule(model, t):
            if t == 0:
                return model.soc[t] == self.initial_soc
            else:
                return (
                    model.soc[t]
                    == model.soc[t - 1] + (self.eta * model.charge[t] - (1.0 / self.eta) * model.discharge[t]) * 1
                )

        model.soc_constr = pyo.Constraint(model.t, rule=soc_rule)

    def battery_energy_schedule(self, time_horizon, delta_t):
        e_el = np.zeros(time_horizon)
        e_el[0] = self.initial_soc
        # Compute battery energy schedule
        for t in range(1, time_horizon):
            # Update energy storage with efficiency factors
            e_el[t] = (
                e_el[t - 1]
                + (self.eta * self.p_el_charge_schedule[t] - (self.p_el_discharge_schedule[t] / self.eta)) * delta_t
            )

            # Apply constraints
            e_el[t] = min(e_el[t], self.max_capacity)  # Ensure max capacity is not exceeded
            e_el[t] = max(e_el[t], 0)
            if e_el[t] == self.max_capacity:
                self.p_el_charge_schedule[t] = 0

            # Prevent discharging if at zero capacity
            if e_el[t] == 0:
                self.p_el_discharge_schedule[t] = 0
        return e_el
