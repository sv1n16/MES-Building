import pyomo.environ as pyo


class Building:
    def __init__(self, building_components=[], heatload=None, time_horizon=12, **kwargs):

        building_parameters = kwargs.get("thermal", {})
        self.building_components = building_components
        self.C, self.U = self.estimate_thermal_parameters(
            floor_area_m2=building_parameters.get("size", 100),
            construction_type=building_parameters.get("construction_type", "medium"),
            insulation_level=building_parameters.get("insulation_level", "average"),
        )
        self.model = kwargs.get("model", None)
        self.time_horizon = kwargs.get("time_horizon", time_horizon)
        self.T_out = kwargs.get("T_out")  # Default outdoor temperature if not provided
        self.T_init = kwargs.get("T_init", 20)  # Default initial indoor temperature if not provided
        self.T_set = kwargs.get("T_set", [21] * time_horizon)  # Default setpoint if not provided
        self.update_model_parameters()
        self.update_model_constraints()
        print(building_components)
        self.heatload = heatload

        if self.heatload is None:
            self.heatload = [0] * time_horizon  # Default heat load if not provided

    def estimate_thermal_parameters(self, floor_area_m2, construction_type="medium", insulation_level="average"):
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

    def set_parameters(self, time_horizon, T_out, T_init, T_set):
        # model.q_heat = pyo.Param(
        #     range(time_horizon),
        #     mutable=True,
        # )
        self.model.T_set = pyo.Param(self.model.t, initialize={t: T_set[t] for t in range(time_horizon)})
        self.model.T_out = pyo.Param(self.model.t, initialize={t: T_out[t] for t in range(time_horizon)})
        self.model.T_in = pyo.Var(self.model.t, within=pyo.NonNegativeReals, initialize=T_init)
        self.model.q_heat = pyo.Var(self.model.t, within=pyo.NonNegativeReals)

    def set_building_constraints(self, model, time_horizon, T_out, T_init, delta_t):

        def heat_balance_rule(model, t):
            return self.model.q_heat_vars[t] + model.q_boiler_vars[t] == model.q_heat[t]

        self.model.heat_demand_match = pyo.Constraint(model.t, rule=heat_balance_rule)

        def thermal_dynamics(self, model, t, T_init, delta_t):
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
                    model.q_heat[t] - self.U * (T_init - model.T_out[t])
                )

            return model.T_in[t] == model.T_in[t - 1] + delta_t / self.C * (
                model.q_heat[t] - self.U * (model.T_in[t - 1] - model.T_out[t])
            )

        self.model.thermal_inertia = pyo.Constraint(
            self.model.t,
            rule=lambda model, t: thermal_dynamics(self, model, t, T_init, delta_t),
        )
        Tmax = 26  # Maximum allowed indoor temperature (°C)

    def set_component_parameters(self):
        for component in self.building_components:
            component.set_parameters(self.model)

    def set_component_constraints(self):
        for component in self.building_components:
            component.set_constraints(self.model)

    def update_model_parameters(self):
        self.set_parameters(
            self.time_horizon,
            self.T_out,
            self.T_init,
            self.T_set,
        )
        self.set_component_parameters()

    def update_model_constraints(self):
        self.set_building_constraints(self.model, self.time_horizon, self.T_out, self.T_init, delta_t=1)
        self.set_component_constraints()

    def get_model(self):
        return self.model
