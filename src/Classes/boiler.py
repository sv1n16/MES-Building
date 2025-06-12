import pyomo.environ as pyo


class GasBoiler:
    """
    Gas Boiler class for optimisation scheduling.
    """

    def __init__(
        self,
        time_horizon,
        max_thermal_power=20.0,  # kW, maximum thermal output
        min_thermal_power=0.0,  # kW, minimum thermal output
        efficiency=0.9,  # Boiler efficiency (fraction)
        gas_price=0.5,  # Gas price (£/kWh or €/kWh)
        p_th_heat=None,  # Thermal output schedule (kW)
    ):
        self.time_horizon = time_horizon
        self.max_thermal_power = max_thermal_power
        self.min_thermal_power = min_thermal_power
        self.efficiency = efficiency
        self.gas_price = gas_price

        # These will be filled by the optimiser
        self.thermal_output_schedule = [0.0] * time_horizon  # kW
        self.gas_consumption_schedule = [0.0] * time_horizon  # kWh
        self.p_th_heat = p_th_heat

    def set_parameters(self, model):
        """
        Set the parameters for the gas boiler in the optimisation model.
        """
        # model.thermal_output = pyo.Var(model.t, bounds=(self.min_thermal_power, self.max_thermal_power), initialize=0)
        model.gas_consumption = pyo.Var(model.t, domain=pyo.Reals, bounds=(0, None), initialize=0)
        model.p_th_boiler_vars = pyo.Var(model.t, bounds=(-self.max_thermal_power, 0), initialize=0)
        model.boiler_on = pyo.Var(model.t, within=pyo.Binary)

    def set_constraints(self, model):
        """
        Set the constraints for the gas boiler in the optimisation model.
        """

        def gas_consumption_rule(model, t):
            return (
                model.gas_consumption[t] == -model.p_th_boiler_vars[t] / self.efficiency
                if self.efficiency > 0
                else 0.0
            )

        model.gas_consumption_constr = pyo.Constraint(model.t, rule=gas_consumption_rule)

        # Boiler output is zero if off, and between -max_thermal_power and 0 if on
        def boiler_output_onoff_rule(model, t):
            return model.p_th_boiler_vars[t] >= -self.max_thermal_power * model.boiler_on[t]

        model.boiler_output_onoff_min = pyo.Constraint(model.t, rule=boiler_output_onoff_rule)

        def boiler_output_onoff_max_rule(model, t):
            return model.p_th_boiler_vars[t] <= 0 * model.boiler_on[t]

        model.boiler_output_onoff_max = pyo.Constraint(model.t, rule=boiler_output_onoff_max_rule)

    def set_thermal_output(self, schedule):
        """
        Set the boiler's thermal output schedule (kW).
        """
        self.thermal_output_schedule = schedule
        self.gas_consumption_schedule = [
            output / self.efficiency if self.efficiency > 0 else 0.0 for output in schedule
        ]

    def get_gas_cost(self):
        """
        Calculate total gas cost for the current schedule.
        """
        return sum(self.gas_consumption_schedule) * self.gas_price
