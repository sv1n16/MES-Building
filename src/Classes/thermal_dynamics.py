# Heat inputs
Q_hp = COP * model.p_hp[t]
Q_boiler = model.p_boiler[t]


# Thermal dynamics
def thermal_inertia(model, t):
    if t == 0:
        return model.T_in[t] == T_init + delta_t / C * (Q_hp + Q_boiler - U * (T_init - T_out[t]))
    return model.T_in[t] == model.T_in[t - 1] + delta_t / C * (Q_hp + Q_boiler - U * (model.T_in[t - 1] - T_out[t]))
