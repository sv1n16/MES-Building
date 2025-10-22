import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Tuple
import pyomo.environ as pyo


@dataclass
class BuildingParams:
    """Parameters for individual building"""

    id: int
    max_power_import: float  # kW
    max_power_export: float  # kW
    battery_capacity: float  # kWh
    battery_max_power: float  # kW
    battery_efficiency: float
    thermal_mass: float  # kWh/K
    heating_power_max: float  # kW
    cop_heatpump: float
    gas_boiler_max: float  # kW
    gas_boiler_eff: float
    comfort_temp_min: float  # °C
    comfort_temp_max: float  # °C


@dataclass
class EnergyData:
    """Energy market and environmental data"""

    electricity_prices: np.ndarray  # £/kWh for each time step
    gas_prices: np.ndarray  # £/kWh for each time step
    outdoor_temp: np.ndarray  # °C for each time step
    solar_forecast: np.ndarray  # kW for each building and time step
    base_load: np.ndarray  # kW for each building and time step


class CentralizedOptimizer:
    """Centralized optimization for multi-building cluster"""

    def __init__(
        self, buildings: List[BuildingParams], energy_data: EnergyData, time_horizon: int = 24, dt: float = 1.0
    ):
        self.buildings = buildings
        self.energy_data = energy_data
        self.n_buildings = len(buildings)
        self.time_horizon = time_horizon
        self.dt = dt  # time step in hours

    def optimize_cluster(self) -> Dict:
        """
        Solve centralized optimization problem for all buildings using Pyomo
        Returns optimal schedules for all buildings
        """
        n_b = self.n_buildings
        n_t = self.time_horizon
        model = pyo.ConcreteModel()
        model.B = pyo.RangeSet(0, n_b - 1)
        model.T = pyo.RangeSet(0, n_t - 1)

        # Decision variables
        model.P_import = pyo.Var(model.B, model.T, within=pyo.NonNegativeReals)
        model.P_export = pyo.Var(model.B, model.T, within=pyo.NonNegativeReals)
        model.P_battery_charge = pyo.Var(model.B, model.T, within=pyo.NonNegativeReals)
        model.P_battery_discharge = pyo.Var(model.B, model.T, within=pyo.NonNegativeReals)
        model.P_heatpump = pyo.Var(model.B, model.T, within=pyo.NonNegativeReals)
        model.P_gas_boiler = pyo.Var(model.B, model.T, within=pyo.NonNegativeReals)
        model.T_indoor = pyo.Var(model.B, model.T)
        model.SOC = pyo.Var(model.B, pyo.RangeSet(0, n_t), within=pyo.NonNegativeReals)
        model.P_share_send = pyo.Var(model.B, model.B, model.T, within=pyo.NonNegativeReals)

        # Auxiliary variables for comfort violations
        model.temp_violation_high = pyo.Var(model.B, model.T, within=pyo.NonNegativeReals)
        model.temp_violation_low = pyo.Var(model.B, model.T, within=pyo.NonNegativeReals)

        # Parameters
        elec_price = self.energy_data.electricity_prices
        gas_price = self.energy_data.gas_prices
        outdoor_temp = self.energy_data.outdoor_temp
        solar_forecast = self.energy_data.solar_forecast
        base_load = self.energy_data.base_load
        dt = self.dt

        # Objective function
        def obj_rule(m):
            electricity_cost = sum(elec_price[t] * m.P_import[b, t] for b in m.B for t in m.T)
            gas_cost = sum(gas_price[t] * m.P_gas_boiler[b, t] for b in m.B for t in m.T)
            export_revenue = sum(0.5 * elec_price[t] * m.P_export[b, t] for b in m.B for t in m.T)
            comfort_penalty = sum(
                100 * (m.temp_violation_high[b, t] + m.temp_violation_low[b, t]) for b in m.B for t in m.T
            )
            return electricity_cost + gas_cost - export_revenue + comfort_penalty

        model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

        # Constraints
        def power_balance_rule(m, b, t):
            return (
                m.P_import[b, t]
                - m.P_export[b, t]
                + solar_forecast[b, t]
                + m.P_battery_discharge[b, t]
                - m.P_battery_charge[b, t]
                + sum(m.P_share_send[bb, b, t] for bb in m.B)
                - sum(m.P_share_send[b, bb, t] for bb in m.B)
                == base_load[b, t] + m.P_heatpump[b, t]
            )

        model.power_balance = pyo.Constraint(model.B, model.T, rule=power_balance_rule)

        # Comfort violation constraints
        def comfort_high_rule(m, b, t):
            return m.temp_violation_high[b, t] >= m.T_indoor[b, t] - self.buildings[b].comfort_temp_max

        model.comfort_high_constr = pyo.Constraint(model.B, model.T, rule=comfort_high_rule)

        def comfort_low_rule(m, b, t):
            return m.temp_violation_low[b, t] >= self.buildings[b].comfort_temp_min - m.T_indoor[b, t]

        model.comfort_low_constr = pyo.Constraint(model.B, model.T, rule=comfort_low_rule)

        # Add other constraints (limits, dynamics, comfort, etc.)
        # ...existing code for constraints, translated to Pyomo...

        # Solve
        solver = pyo.SolverFactory("gurobi_direct")
        result = solver.solve(model, tee=True)

        # Extract results
        results = {
            "status": str(result.solver.status),
            "total_cost": pyo.value(model.obj),
            "P_import": np.array([[pyo.value(model.P_import[b, t]) for t in model.T] for b in model.B]),
            "P_export": np.array([[pyo.value(model.P_export[b, t]) for t in model.T] for b in model.B]),
            "P_battery_charge": np.array(
                [[pyo.value(model.P_battery_charge[b, t]) for t in model.T] for b in model.B]
            ),
            "P_battery_discharge": np.array(
                [[pyo.value(model.P_battery_discharge[b, t]) for t in model.T] for b in model.B]
            ),
            "P_heatpump": np.array([[pyo.value(model.P_heatpump[b, t]) for t in model.T] for b in model.B]),
            "P_gas_boiler": np.array([[pyo.value(model.P_gas_boiler[b, t]) for t in model.T] for b in model.B]),
            "T_indoor": np.array([[pyo.value(model.T_indoor[b, t]) for t in model.T] for b in model.B]),
            "SOC": np.array([[pyo.value(model.SOC[b, t]) for t in range(n_t + 1)] for b in model.B]),
            "P_share_send": np.array(
                [[[pyo.value(model.P_share_send[b1, b2, t]) for t in model.T] for b2 in model.B] for b1 in model.B]
            ),
        }
        return results

    def plot_results(self, results: Dict):
        """Plot optimization results"""
        if results is None:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        time_hours = np.arange(self.time_horizon)

        # Plot 1: Power consumption and generation
        ax1 = axes[0, 0]
        for b in range(min(3, self.n_buildings)):  # Show first 3 buildings
            ax1.plot(time_hours, results["P_import"][b, :], label=f"Building {b+1} Import", linestyle="-")
            ax1.plot(time_hours, -results["P_export"][b, :], label=f"Building {b+1} Export", linestyle="--")
        ax1.set_xlabel("Time (hours)")
        ax1.set_ylabel("Power (kW)")
        ax1.set_title("Grid Power Import/Export")
        ax1.legend()
        ax1.grid(True)

        # Plot 2: Battery operation
        ax2 = axes[0, 1]
        for b in range(min(3, self.n_buildings)):
            ax2.plot(time_hours, results["SOC"][b, :-1], label=f"Building {b+1} SOC")
        ax2.set_xlabel("Time (hours)")
        ax2.set_ylabel("State of Charge (kWh)")
        ax2.set_title("Battery State of Charge")
        ax2.legend()
        ax2.grid(True)

        # Plot 3: Indoor temperatures
        ax3 = axes[1, 0]
        for b in range(min(3, self.n_buildings)):
            ax3.plot(time_hours, results["T_indoor"][b, :], label=f"Building {b+1}")
            ax3.axhline(y=self.buildings[b].comfort_temp_min, color="red", linestyle=":", alpha=0.5)
            ax3.axhline(y=self.buildings[b].comfort_temp_max, color="red", linestyle=":", alpha=0.5)
        ax3.plot(time_hours, self.energy_data.outdoor_temp, "k--", label="Outdoor", alpha=0.7)
        ax3.set_xlabel("Time (hours)")
        ax3.set_ylabel("Temperature (°C)")
        ax3.set_title("Indoor Temperatures")
        ax3.legend()
        ax3.grid(True)

        # Plot 4: Energy prices and heating power
        ax4 = axes[1, 1]
        ax4_twin = ax4.twinx()

        ax4.plot(time_hours, self.energy_data.electricity_prices, "b-", label="Electricity Price")
        ax4.plot(time_hours, self.energy_data.gas_prices, "r-", label="Gas Price")
        ax4.set_xlabel("Time (hours)")
        ax4.set_ylabel("Price (£/kWh)", color="black")
        ax4.legend(loc="upper left")

        total_heating = results["P_heatpump"][0, :] + results["P_gas_boiler"][0, :]
        ax4_twin.plot(time_hours, total_heating, "g-", label="Heating Power (B1)")
        ax4_twin.set_ylabel("Power (kW)", color="green")
        ax4_twin.legend(loc="upper right")

        ax4.set_title("Energy Prices and Heating Schedule")
        ax4.grid(True)

        plt.tight_layout()
        plt.show()


# Example usage and test case
def create_test_scenario():
    """Create test scenario with sample buildings and data"""

    # Create sample buildings
    buildings = [
        BuildingParams(
            id=1,
            max_power_import=10,
            max_power_export=5,
            battery_capacity=20,
            battery_max_power=5,
            battery_efficiency=0.9,
            thermal_mass=50,
            heating_power_max=8,
            cop_heatpump=3.0,
            gas_boiler_max=15,
            gas_boiler_eff=0.9,
            comfort_temp_min=19,
            comfort_temp_max=23,
        ),
        BuildingParams(
            id=2,
            max_power_import=8,
            max_power_export=4,
            battery_capacity=15,
            battery_max_power=4,
            battery_efficiency=0.9,
            thermal_mass=40,
            heating_power_max=6,
            cop_heatpump=2.8,
            gas_boiler_max=12,
            gas_boiler_eff=0.85,
            comfort_temp_min=20,
            comfort_temp_max=24,
        ),
        BuildingParams(
            id=3,
            max_power_import=12,
            max_power_export=6,
            battery_capacity=25,
            battery_max_power=6,
            battery_efficiency=0.9,
            thermal_mass=60,
            heating_power_max=10,
            cop_heatpump=3.2,
            gas_boiler_max=18,
            gas_boiler_eff=0.92,
            comfort_temp_min=18,
            comfort_temp_max=22,
        ),
    ]

    # Create sample energy data (24-hour horizon)
    time_horizon = 24
    n_buildings = len(buildings)

    # Dynamic electricity pricing (higher during peak hours)
    elec_prices = np.array(
        [
            0.10,
            0.09,
            0.08,
            0.08,
            0.09,
            0.12,
            0.18,
            0.22,
            0.25,
            0.20,
            0.15,
            0.14,
            0.13,
            0.14,
            0.16,
            0.20,
            0.28,
            0.35,
            0.32,
            0.25,
            0.20,
            0.15,
            0.12,
            0.10,
        ]
    )

    # Gas prices (more stable, daily average)
    gas_prices = np.full(time_horizon, 0.06)

    # Outdoor temperature (winter day)
    outdoor_temp = np.array([2, 1, 0, -1, 0, 1, 2, 4, 6, 8, 10, 12, 13, 14, 13, 12, 10, 8, 6, 5, 4, 3, 3, 2])

    # Solar generation (typical winter day with PV)
    solar_forecast = np.zeros((n_buildings, time_horizon))
    solar_profile = np.array(
        [0, 0, 0, 0, 0, 0, 0.5, 1.5, 3.0, 4.5, 5.5, 6.0, 6.2, 6.0, 5.0, 3.5, 2.0, 0.8, 0, 0, 0, 0, 0, 0]
    )

    for b in range(n_buildings):
        # Different PV capacity for each building
        pv_capacity = [4, 3, 5][b]  # kW
        solar_forecast[b, :] = solar_profile * pv_capacity / 6.2  # Normalize

    # Base electrical load (without heating)
    base_load = np.zeros((n_buildings, time_horizon))
    load_profile = np.array(
        [
            1.5,
            1.2,
            1.0,
            1.0,
            1.2,
            2.0,
            3.5,
            4.0,
            3.0,
            2.5,
            2.0,
            2.2,
            2.5,
            2.3,
            2.0,
            2.5,
            4.0,
            5.5,
            4.8,
            3.5,
            3.0,
            2.5,
            2.0,
            1.8,
        ]
    )

    for b in range(n_buildings):
        # Different load patterns
        scale_factor = [1.0, 0.8, 1.2][b]
        base_load[b, :] = load_profile * scale_factor

    energy_data = EnergyData(
        electricity_prices=elec_prices,
        gas_prices=gas_prices,
        outdoor_temp=outdoor_temp,
        solar_forecast=solar_forecast,
        base_load=base_load,
    )

    return buildings, energy_data


if __name__ == "__main__":
    # Create test scenario
    buildings, energy_data = create_test_scenario()

    # Initialize optimizer
    optimizer = CentralizedOptimizer(buildings, energy_data, time_horizon=24)

    # Run optimization
    print("Running centralized optimization for 3-building cluster...")
    results = optimizer.optimize_cluster()

    if results:
        print(f"Optimization successful!")
        print(f"Total cost: £{results['total_cost']:.2f}")
        print(f"Status: {results['status']}")

        # Plot results
        optimizer.plot_results(results)

        # Print detailed summary statistics
        print("\n=== Summary Statistics ===")
        total_import = np.sum(results["P_import"])
        total_export = np.sum(results["P_export"])
        total_solar = np.sum(energy_data.solar_forecast)
        total_heating = np.sum(results["P_heatpump"]) + np.sum(results["P_gas_boiler"])

        print(f"Total grid import: {total_import:.2f} kWh")
        print(f"Total grid export: {total_export:.2f} kWh")
        print(f"Total solar generation: {total_solar:.2f} kWh")
        print(f"Net grid consumption: {total_import - total_export:.2f} kWh")
        print(f"Total heating energy: {total_heating:.2f} kWh")

        # Inter-building energy sharing analysis
        total_sharing = np.sum(results["P_share_send"])
        print(f"Total inter-building energy sharing: {total_sharing:.2f} kWh")

        # Check comfort violations
        for b, building in enumerate(buildings):
            high_violations = np.sum(results["T_indoor"][b, :] > building.comfort_temp_max)
            low_violations = np.sum(results["T_indoor"][b, :] < building.comfort_temp_min)
            total_violations = high_violations + low_violations
            print(f"Building {b+1} comfort violations: {total_violations} hours")

        # Energy source breakdown
        print("\n=== Energy Source Breakdown ===")
        for b in range(len(buildings)):
            heatpump_energy = np.sum(results["P_heatpump"][b, :])
            gas_energy = np.sum(results["P_gas_boiler"][b, :])
            total_heating_b = heatpump_energy + gas_energy
            if total_heating_b > 0:
                hp_percentage = (heatpump_energy / total_heating_b) * 100
                gas_percentage = (gas_energy / total_heating_b) * 100
                print(f"Building {b+1}: {hp_percentage:.1f}% heat pump, {gas_percentage:.1f}% gas boiler")
    else:
        print("Optimization failed!")
