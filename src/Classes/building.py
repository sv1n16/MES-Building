import pyomo.environ as pyo


class Building:
    def __init__(
        self, building_size=100, building_components=[], construction_type="medium", insulation_level="average"
    ):
        self.building_components = building_components
        self.C, self.U = self.estimate_thermal_parameters(
            floor_area_m2=building_size,
            construction_type=construction_type,
            insulation_level=insulation_level,
        )

    def estimate_thermal_parameters(floor_area_m2, construction_type="medium", insulation_level="average"):
        """
        Estimate thermal capacity (C) and conductance (U) for a building.

        Parameters:
        - floor_area_m2: float – floor area in m²
        - construction_type: str – 'light', 'medium', or 'heavy'
        - insulation_level: str – 'good', 'average', or 'poor'

        Returns:
        - C (thermal capacity) in kWh/°C
        - U (thermal conductance) in kW/°C
        """

        # Thermal capacity per m² by construction type [kWh/°C per m²]
        c_lookup = {
            "light": 0.01,  # 1 kWh/°C per 100 m²
            "medium": 0.1,  # 10 kWh/°C per 100 m²
            "heavy": 0.2,  # 20 kWh/°C per 100 m²
        }

        # Thermal conductance per m² by insulation quality [kW/°C per 100 m²]
        u_lookup = {"good": 0.2 / 100, "average": 0.5 / 100, "poor": 1.0 / 100}

        if construction_type not in c_lookup:
            raise ValueError("Invalid construction type. Choose from: 'light', 'medium', 'heavy'")
        if insulation_level not in u_lookup:
            raise ValueError("Invalid insulation level. Choose from: 'good', 'average', 'poor'")

        C = c_lookup[construction_type] * floor_area_m2  # kWh/°C
        U = u_lookup[insulation_level] * floor_area_m2  # kW/°C

        return round(C, 2), round(U, 3)

    def set_building_constraits(self, model, time_horizon, T_out, T_init, delta_t, COP):
        def thermal_dymanics(self, model, t, T_out, T_init, delta_t, C, U, COP):
            """
            Define the thermal dynamics of the building.

            Parameters:
            - model: Pyomo model object
            - t: time step
            - T_out: outdoor temperature at time t
            - T_init: initial indoor temperature
            - delta_t: time step duration in hours
            - C: thermal capacity of the building in kWh/°C
            - U: thermal conductance of the building in kW/°C
            - COP: Coefficient of Performance for the heat pump

            Returns:
            - Pyomo constraint for thermal inertia
            """

            if t == 0:
                return model.T_in[t] == T_init + delta_t / self.C * (
                    COP * model.p_hp[t] + model.p_boiler[t] - self.U * (T_init - T_out[t])
                )

            return model.T_in[t] == model.T_in[t - 1] + delta_t / self.C * (
                COP * model.p_hp[t] + model.p_boiler[t] - self.U * (model.T_in[t - 1] - T_out[t])
            )

        model.thermal_inertia = pyo.Constraint(
            model.t,
            rule=lambda model, t: thermal_dymanics(self, model, t, T_out, T_init, delta_t, C, U, COP),
        )

        def q_in_rule(model, t):
            """
            Calculate the heat input to the building at time t.

            Parameters:
            - model: Pyomo model object
            - t: time step

            Returns:
            - Pyomo expression for heat input
            """
            return model.q_in[t] == COP * model.p_hp[t] + model.p_boiler[t]

        model.q_in = pyo.Constraint(model.t, rule=q_in_rule)

        def q_in_rule_update(model, t):
            model.q_in[t] = (model.T_in[t] - model.T_out[t]) * self.U + self.C * (
                model.T_in[t] - model.T_in[t - 1]
            ) / delta_t

        model.q_in_update = pyo.Constraint(model.t, rule=q_in_rule_update)

    def set_parameters(self, model, time_horizon, T_out, T_init, delta_t, C, U, COP, T_setpoint):
        """
        Set the thermal output of the building based on the heat pump and boiler outputs.

        Parameters:
        - model: Pyomo model object
        - time_horizon: total number of time steps
        - T_out: outdoor temperature at each time step
        - T_init: initial indoor temperature
        - delta_t: time step duration in hours
        - C: thermal capacity of the building in kWh/°C
        - U: thermal conductance of the building in kW/°C
        - COP: Coefficient of Performance for the heat pump
        """
        model.T_in = pyo.Var(range(time_horizon), within=pyo.NonNegativeReals, initialize=T_init)
        model.T_out = pyo.Param(range(time_horizon), initialize=lambda t: T_out[t], mutable=True)
        model.T_setpoint = pyo.Param(range(time_horizon), initialize=lambda t: T_setpoint[t], mutable=True)
        model.q_in = pyo.Var(range(time_horizon), within=pyo.NonNegativeReals)
