import numpy as np 
import pandas as pd 

class PVModule:

    def __init__(self, time_horizon,start_point, radiation, method=1, area=0.0, peak_power=0.0, eta_noct=0.18, radiation_noct=1000.0,
                 t_cell_noct=45.0, t_ambient_noct=20.0, alpha_noct=0, beta=0, gamma=0, tau_alpha=0.9,
                 force_renewables=True):
        
        """
        method: how power is computed, 1 computes power based on peak power
        area : float, optional
            PV unit installation area in m^2
            Requires ``method=0``
        peak_power : float, optional
            PV peak power installation in kWp
            Requires ``method=1``
        eta_noct : float, optional
            Electrical efficiency at NOCT conditions (without unit)
            NOCT conditions: See manufacturer's data sheets or
            Duffie, Beckman - Solar Engineering of Thermal Processes (4th ed.), page 759
            Requires ``method=0``
        radiation_noct : float, optional
            Nominal solar radiation at NOCT conditions (in W/m^2)
            NOCT conditions: See manufacturer's data sheets or
            Duffie, Beckman - Solar Engineering of Thermal Processes (4th ed.), page 759
        t_cell_noct : float, optional
            Nominal cell temperature at NOCT conditions (in degree Celsius)
            NOCT conditions: See manufacturer's data sheets or
            Duffie, Beckman - Solar Engineering of Thermal Processes (4th ed.), page 759
        t_ambient_noct : float, optional
            Nominal ambient air temperature at NOCT conditions (in degree Celsius)
            NOCT conditions: See manufacturer's data sheets or
            Duffie, Beckman - Solar Engineering of Thermal Processes (4th ed.), page 759
        alpha_noct : float, optional
            Temperature coefficient at NOCT conditions (without unit)
            NOCT conditions: See manufacturer's data sheets or
            Duffie, Beckman - Solar Engineering of Thermal Processes (4th ed.), page 759
        beta : float, optional
            Slope, the angle (in degree) between the plane of the surface in 
            question and the horizontal. 0 <= beta <= 180. If beta > 90, the 
            surface faces downwards.
        gamma : float, optional
            Surface azimuth angle. The deviation of the projection on a 
            horizontal plane of the normal to the surface from the local 
            meridian, with zero due south, east negative, and west positive.
            -180 <= gamma <= 180
        tau_alpha : float
            Optical properties of the PV unit. Product of absorption and 
            transmission coeffients.
            According to Duffie, Beckman - Solar Engineering of Thermal 
            Processes (4th ed.), page 758, this value is typically close to 0.9
        """
        self._kind = "pv"        
        
        self.time_horizon = time_horizon
        self.radiation = radiation
        self.method = method

        self.area = area
        self.peak_power = peak_power

        self.eta_noct = eta_noct
        self.radiation_noct = radiation_noct
        self.t_cell_noct = t_cell_noct
        self.t_ambient_noct = t_ambient_noct
        self.alpha_noct = alpha_noct

        self.beta = beta
        self.gamma = gamma
        self.tau_alpha = tau_alpha
        
        self.total_power = np.zeros(time_horizon)
        self.total_radiation = np.zeros(time_horizon)
        self.current_power = np.zeros(time_horizon)

        self.force_renewables = force_renewables
        self.getPower(currentValues=False) #calculates total power for the entire simulation horizon
        print(len(self.total_power))
        self.p_el_supply = self.total_power[start_point:start_point+self.time_horizon] / 1000 
        
    def _computePowerArea(self, currentValues=True):
        """
        Compute PV electric output power based on a certain area equipped with PV panels

        Parameters
        ----------
        currentValues : bool, optional
            If True, returns values of current horizon (default: True).
            If False, returns annual values.

        Returns
        -------
        res_tuple : tuple
            2d tuple holding power array in Watt and radiation array in W/m^2
        """
        # Get radiation on a tilted surface


        # If no temperature coefficient is given, a simple PV model is applied
        if self.alpha_noct == 0:
            power = self.area * self.eta_noct * self.radiation
        else:
            # Get ambient temperature
            getTemperature = self.environment.weather.getWeatherForecast
            t_ambient = getTemperature(getTAmbient=True, currentValues=currentValues)

            # Compute the cell temperature. 
            # Assumption: Wind velocity is 1 m/s (same as NOCT conditions)
            # The resulting equation is based on equation 23.3.3 (page 758,
            # Duffie, Beckman - Solar Engineering of Thermal Processes, 4th ed)
            # as well as equation 3 (Skroplaki, Palyvos - 2009 - On the 
            # temperature dependence of photovoltaic module electrical 
            # performance. A review of efficiency-power correlations.)
            
            # Introduce a few abbreviations
            a1 = (self.t_cell_noct - self.t_ambient_noct) * self.radiation[0] / self.radiation_noct
            denominator = 1 - a1 * self.alpha_noct * self.eta_noct / self.tau_alpha
            numerator = 1 - self.alpha_noct * (t_ambient[0] - self.t_cell_noct + a1)
            eta = self.eta_noct * numerator / denominator
        
            # Compute power
            power = self.area * eta * self.radiation
        
        return (power, self.radiation[0])
    def getPower(self, time_horizon=None, currentValues=True,current_timestep=0, timesteps=1):
        """
        compute the power over the simulation horizon
        """
        (current_power, currentRadiation) = self._computePowerArea(currentValues=currentValues)
        if currentValues: 
            self.current_power = current_power

            self.total_power[current_timestep:(current_timestep + timesteps)] = current_power
            self.total_radiation[current_timestep:(current_timestep + timesteps)] = currentRadiation
            return self.current_power

        else:
            self.total_power = current_power
            self.total_radiation = currentRadiation
            return self.total_power

