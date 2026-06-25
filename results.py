import argparse
import json
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── colour palette ────────────────────────────────────────────────────────────
OL_COLOR = "#378ADD"  # blue  – open loop
CO_COLOR = "#D85A30"  # coral – central optimisation
SP_COLOR = "#888780"  # grey  – setpoint / reference
OCC_FILL = "rgba(232,168,124,0.20)"  # warm tint – occupied hours
DB_FILL = "rgba(55,138,221,0.10)"  # blue tint  – deadband band
DB_LINE = "rgba(55,138,221,0.55)"  # blue line  – deadband edges
VIOL_COLOR = "#A32D2D"  # dark red   – deadband violations

HOURS = list(range(24))

# ── academic typography ───────────────────────────────────────────────────────
FONT_FAMILY = "Aptos"
FONT_SIZE_TITLE = 16
FONT_SIZE_AXIS = 14
FONT_SIZE_TICK = 14
FONT_SIZE_ANNOTATION = 14
AXIS_STANDOFF = 8


# ── helpers ───────────────────────────────────────────────────────────────────


def schedule(inner, building, key, fallback=None):
    """Return a 24-element list for *key* from *building*'s schedules.
    If the key is absent, returns *fallback* (default: 24 × None)."""
    v = inner[building]["schedules"].get(key, fallback)
    if v is None:
        return [None] * 24
    return v if isinstance(v, list) else [v] * 24


def peak(inner, building, key):
    return inner[building]["peak_loads"][key]


def short(building_key: str) -> str:
    return building_key.replace(" electricity consumption", "")


def common_layout(**kwargs) -> dict:
    base = dict(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family=FONT_FAMILY, size=FONT_SIZE_AXIS),
        margin=dict(t=90, b=60, l=80, r=80),
    )
    base.update(kwargs)
    return base


def legend_top() -> dict:
    return dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
        font=dict(family=FONT_FAMILY, size=FONT_SIZE_TICK),
    )


def legend_2x2() -> dict:
    """Legend for 2×2 subplots: centred below the plots, clear of the title."""
    return dict(
        orientation="h",
        yanchor="top",
        y=-0.08,
        xanchor="center",
        x=0.5,
        font=dict(family=FONT_FAMILY, size=FONT_SIZE_TICK),
    )


def academic_xaxis(**kwargs) -> dict:
    base = dict(
        title_font=dict(family=FONT_FAMILY, size=FONT_SIZE_AXIS),
        title_standoff=AXIS_STANDOFF,
        tickfont=dict(family=FONT_FAMILY, size=FONT_SIZE_TICK),
        gridcolor="lightgrey",
        linecolor="black",
        linewidth=0.8,
        mirror=True,
        automargin=True,
    )
    base.update(kwargs)
    return base


def academic_yaxis(**kwargs) -> dict:
    base = dict(
        title_font=dict(family=FONT_FAMILY, size=FONT_SIZE_AXIS),
        title_standoff=AXIS_STANDOFF,
        tickfont=dict(family=FONT_FAMILY, size=FONT_SIZE_TICK),
        gridcolor="lightgrey",
        linecolor="black",
        linewidth=0.8,
        mirror=True,
        automargin=True,
    )
    base.update(kwargs)
    return base


## Gas boiler

BOILER_EFFICIENCY = 0.9  # condensing gas boiler; override if your model uses a different value


def gas_import(inner, building) -> list:
    """Derive hourly gas import (kWh) from boiler thermal output and efficiency."""
    thermal = schedule(inner, building, "boiler_thermal_output")
    return [v / BOILER_EFFICIENCY if v else 0.0 for v in thermal]


def plot_gas_import(inner1, inner2, buildings, labels):
    """2×2 hourly gas import profiles with electricity price on secondary y."""
    gas_ol = [gas_import(inner1, b) for b in buildings]
    gas_co = [gas_import(inner2, b) for b in buildings]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[f"{sl} – hourly gas use" for sl in labels],
        specs=[
            [{"secondary_y": True}, {"secondary_y": True}],
            [{"secondary_y": True}, {"secondary_y": True}],
        ],
        vertical_spacing=0.22,
        horizontal_spacing=0.22,
    )

    panel_positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

    for idx, (b, sl) in enumerate(zip(buildings, labels)):
        row, col = panel_positions[idx]
        ep = schedule(inner2, b, "electricity_price")
        show = idx == 0

        fig.add_trace(
            go.Bar(
                name="Open loop",
                x=HOURS,
                y=gas_ol[idx],
                marker_color=OL_COLOR,
                opacity=0.80,
                legendgroup="ol",
                showlegend=show,
            ),
            row=row, col=col, secondary_y=False,
        )

        fig.add_trace(
            go.Bar(
                name="Central optimisation",
                x=HOURS,
                y=gas_co[idx],
                marker_color=CO_COLOR,
                opacity=0.80,
                legendgroup="co",
                showlegend=show,
            ),
            row=row, col=col, secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                name="Electricity price" if show else None,
                x=HOURS,
                y=ep,
                mode="lines",
                line=dict(color=SP_COLOR, width=1.5, dash="dash"),
                legendgroup="price",
                showlegend=show,
            ),
            row=row, col=col, secondary_y=True,
        )

    for row, col in panel_positions:
        fig.update_yaxes(
            **academic_yaxis(title_text="Gas import (kWh)"),
            secondary_y=False, row=row, col=col,
        )
        fig.update_yaxes(
            **academic_yaxis(title_text="Electricity price (£/kWh)", showgrid=False),
            secondary_y=True, row=row, col=col,
        )
        fig.update_xaxes(**academic_xaxis(title_text="Hour of day"), row=row, col=col)

    fig.update_layout(
        barmode="group",
        title_text="Gas import timing – open loop vs central optimisation",
        title_x=0.5,
        title_xanchor="center",
        title_font=dict(family=FONT_FAMILY, size=FONT_SIZE_TITLE),
        legend=legend_2x2(),
        width=700,
        height=640,
        **common_layout(margin=dict(t=60, b=100, l=80, r=90)),
    )
    for annotation in fig["layout"]["annotations"]:
        annotation["font"] = dict(family=FONT_FAMILY, size=FONT_SIZE_ANNOTATION)
    return fig


# ── occupied-hour shading helper ──────────────────────────────────────────────


def occupancy_shapes(occ: list, y0: float, y1: float) -> list:
    shapes = []
    for h, o in enumerate(occ):
        if o:
            shapes.append(
                dict(
                    type="rect",
                    xref="x",
                    yref="y",
                    x0=h - 0.5,
                    x1=h + 0.5,
                    y0=y0,
                    y1=y1,
                    fillcolor=OCC_FILL,
                    line_width=0,
                    layer="below",
                )
            )
    return shapes


# ── plot 1 – total cost ───────────────────────────────────────────────────────


def plot_cost(inner1, inner2, buildings, labels):
    ol_costs = [peak(inner1, b, "total_cost_gbp") for b in buildings]
    co_costs = [peak(inner2, b, "total_cost_gbp") for b in buildings]

    fig = go.Figure(
        [
            go.Bar(
                name="Open loop",
                x=labels,
                y=ol_costs,
                marker_color=OL_COLOR,
                text=[f"£{v:.2f}" for v in ol_costs],
                textposition="outside",
                textfont=dict(family=FONT_FAMILY, size=FONT_SIZE_TICK),
            ),
            go.Bar(
                name="Central optimisation",
                x=labels,
                y=co_costs,
                marker_color=CO_COLOR,
                text=[f"£{v:.2f}" for v in co_costs],
                textposition="outside",
                textfont=dict(family=FONT_FAMILY, size=FONT_SIZE_TICK),
            ),
        ]
    )
    fig.update_layout(
        barmode="group",
        title_text="Total electricity cost (£)",
        title_x=0.5,
        title_xanchor="center",
        title_font=dict(family=FONT_FAMILY, size=FONT_SIZE_TITLE),
        yaxis=academic_yaxis(title_text="Cost (£)", tickprefix="£", tickformat=".2f"),
        xaxis=academic_xaxis(automargin=True),
        legend=legend_top(),
        width=700,
        height=420,
        **common_layout(),
    )
    return fig


# ── plot 2 – thermal setpoint deviation ──────────────────────────────────────


def plot_temp_deviation(inner1, inner2, buildings, labels):
    ol_dev = [peak(inner1, b, "average_thermal_setpoint_deviation_celsius") for b in buildings]
    co_dev = [peak(inner2, b, "average_thermal_setpoint_deviation_celsius") for b in buildings]

    fig = go.Figure(
        [
            go.Bar(
                name="Open loop",
                x=labels,
                y=ol_dev,
                marker_color=OL_COLOR,
                text=[f"{v:.3f}°C" for v in ol_dev],
                textposition="outside",
                textfont=dict(family=FONT_FAMILY, size=FONT_SIZE_TICK),
            ),
            go.Bar(
                name="Central optimisation",
                x=labels,
                y=co_dev,
                marker_color=CO_COLOR,
                text=[f"{v:.3f}°C" for v in co_dev],
                textposition="outside",
                textfont=dict(family=FONT_FAMILY, size=FONT_SIZE_TICK),
            ),
        ]
    )
    fig.update_layout(
        barmode="group",
        title_text="Average thermal setpoint deviation (°C)",
        title_x=0.5,
        title_xanchor="center",
        title_font=dict(family=FONT_FAMILY, size=FONT_SIZE_TITLE),
        yaxis=academic_yaxis(title_text="Deviation (°C)", tickformat=".3f"),
        xaxis=academic_xaxis(automargin=True),
        legend=legend_top(),
        width=700,
        height=420,
        **common_layout(),
    )
    return fig


# ── plot 3 – peak demand + time of peak ──────────────────────────────────────


def plot_peak_summary(inner1, inner2, buildings, labels):
    ol_peak = [peak(inner1, b, "peak_electricity_demand_kw") for b in buildings]
    co_peak = [peak(inner2, b, "peak_electricity_demand_kw") for b in buildings]
    ol_ptime = [peak(inner1, b, "time_of_peak_electricity_demand_hour") for b in buildings]
    co_ptime = [peak(inner2, b, "time_of_peak_electricity_demand_hour") for b in buildings]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Peak electricity demand (kW)", "Time of peak demand (hour)"),
        horizontal_spacing=0.22,
    )

    for name, color, kw_data, t_data, show in [
        ("Open loop", OL_COLOR, ol_peak, ol_ptime, True),
        ("Central optimisation", CO_COLOR, co_peak, co_ptime, True),
    ]:
        fig.add_trace(
            go.Bar(
                name=name,
                x=labels,
                y=kw_data,
                marker_color=color,
                offsetgroup=name,
                text=[f"{v:.1f}" for v in kw_data],
                textposition="outside",
                textfont=dict(family=FONT_FAMILY, size=FONT_SIZE_TICK),
                showlegend=show,
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Bar(
                name=name,
                x=labels,
                y=t_data,
                marker_color=color,
                offsetgroup=name,
                text=[f"{int(v):02d}:00" for v in t_data],
                textposition="outside",
                textfont=dict(family=FONT_FAMILY, size=FONT_SIZE_TICK),
                showlegend=False,
            ),
            row=1, col=2,
        )

    fig.update_layout(
        barmode="group",
        title_text="Peak demand comparison",
        title_x=0.5,
        title_xanchor="center",
        title_font=dict(family=FONT_FAMILY, size=FONT_SIZE_TITLE),
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="left", x=0,
                    font=dict(family=FONT_FAMILY, size=FONT_SIZE_TICK)),
        width=700,
        height=420,
        **common_layout(),
    )
    for annotation in fig["layout"]["annotations"]:
        annotation["font"] = dict(family=FONT_FAMILY, size=FONT_SIZE_ANNOTATION)
    fig.update_yaxes(**academic_yaxis(title_text="Peak demand (kW)"), row=1, col=1)
    fig.update_yaxes(**academic_yaxis(title_text="Hour of day"), row=1, col=2)
    fig.update_xaxes(**academic_xaxis(automargin=True))
    return fig


# ── plot 4 – hourly grid import (2×2, secondary y = price) ───────────────────


def plot_grid_import(inner1, inner2, buildings, labels):
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[f"{sl} – grid import" for sl in labels],
        specs=[
            [{"secondary_y": True}, {"secondary_y": True}],
            [{"secondary_y": True}, {"secondary_y": True}],
        ],
        vertical_spacing=0.22,
        horizontal_spacing=0.22,
    )

    for idx, (b, sl) in enumerate(zip(buildings, labels)):
        row, col = divmod(idx, 2)
        row += 1
        col += 1

        gi_ol = schedule(inner1, b, "grid_import_schedule")
        gi_co = schedule(inner2, b, "grid_import_schedule")
        ep = schedule(inner2, b, "electricity_price")

        show = idx == 0
        fig.add_trace(
            go.Scatter(
                name="Open loop",
                x=HOURS, y=gi_ol,
                mode="lines+markers",
                line=dict(color=OL_COLOR, width=1.8),
                marker=dict(size=3),
                legendgroup="ol",
                showlegend=show,
            ),
            row=row, col=col, secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                name="Central optimisation",
                x=HOURS, y=gi_co,
                mode="lines+markers",
                line=dict(color=CO_COLOR, width=1.8),
                marker=dict(size=3),
                legendgroup="co",
                showlegend=show,
            ),
            row=row, col=col, secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                name="Electricity price",
                x=HOURS, y=ep,
                mode="lines",
                line=dict(color=SP_COLOR, width=1.5, dash="dash"),
                legendgroup="price",
                showlegend=show,
            ),
            row=row, col=col, secondary_y=True,
        )

    # Apply axis styling to all subplots
    for idx in range(4):
        row, col = divmod(idx, 2)
        row += 1
        col += 1
        fig.update_yaxes(
            **academic_yaxis(title_text="Grid import (kW)"),
            secondary_y=False, row=row, col=col,
        )
        fig.update_yaxes(
            **academic_yaxis(title_text="Price (£/kWh)", showgrid=False),
            secondary_y=True, row=row, col=col,
        )
        fig.update_xaxes(**academic_xaxis(title_text="Hour of day"), row=row, col=col)

    fig.update_layout(
        title_text="Hourly grid import (kW) vs electricity price",
        title_x=0.5,
        title_xanchor="center",
        title_font=dict(family=FONT_FAMILY, size=FONT_SIZE_TITLE),
        legend=legend_2x2(),
        width=700,
        height=640,
        **common_layout(margin=dict(t=60, b=100, l=80, r=90)),
    )
    for annotation in fig["layout"]["annotations"]:
        annotation["font"] = dict(family=FONT_FAMILY, size=FONT_SIZE_ANNOTATION)
    return fig


# ── plot 5 – temperature with deadbands & occupancy, CO only (2×2) ───────────


def plot_temperature_co(inner2, buildings, labels):
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[f"{sl} – indoor temperature (central opt)" for sl in labels],
        vertical_spacing=0.22,
        horizontal_spacing=0.18,
    )

    for idx, (b, sl) in enumerate(zip(buildings, labels)):
        row, col = divmod(idx, 2)
        row += 1
        col += 1

        temp = schedule(inner2, b, "indoor_temperature")
        setpt = schedule(inner2, b, "temperature_setpoint")
        lower = schedule(inner2, b, "temperature_lower_deadband_celsius")
        upper = schedule(inner2, b, "temperature_upper_deadband_celsius")
        occ = schedule(inner2, b, "occupancy_profile")

        y_min = min(lower) - 1
        y_max = max(upper) + 1

        for shape in occupancy_shapes(occ, y_min, y_max):
            shape.update(xref=f"x{idx+1}", yref=f"y{idx+1}")
            fig.add_shape(shape)

        show = idx == 0

        fig.add_trace(
            go.Scatter(
                name="Upper deadband",
                x=HOURS, y=upper,
                mode="lines",
                line=dict(color=DB_LINE, width=1, dash="dot"),
                legendgroup="db_upper",
                showlegend=show,
            ),
            row=row, col=col,
        )

        fig.add_trace(
            go.Scatter(
                name="Deadband range",
                x=HOURS, y=lower,
                mode="lines",
                line=dict(color=DB_LINE, width=1, dash="dot"),
                fill="tonexty",
                fillcolor=DB_FILL,
                legendgroup="db_lower",
                showlegend=show,
            ),
            row=row, col=col,
        )

        fig.add_trace(
            go.Scatter(
                name="Setpoint",
                x=HOURS, y=setpt,
                mode="lines",
                line=dict(color=SP_COLOR, width=1.5, dash="dash"),
                legendgroup="setpt",
                showlegend=show,
            ),
            row=row, col=col,
        )

        violations = [t < lo or t > hi for t, lo, hi in zip(temp, lower, upper)]
        marker_colors = [VIOL_COLOR if v else CO_COLOR for v in violations]

        fig.add_trace(
            go.Scatter(
                name="Indoor temp (central opt)",
                x=HOURS, y=temp,
                mode="lines+markers",
                line=dict(color=CO_COLOR, width=1.8),
                marker=dict(size=[5 if v else 3 for v in violations], color=marker_colors),
                legendgroup="co_temp",
                showlegend=show,
            ),
            row=row, col=col,
        )

    fig.update_layout(
        title_text="Central optimisation – indoor temperature, setpoint, deadband and occupancy",
        title_x=0.5,
        title_xanchor="center",
        title_font=dict(family=FONT_FAMILY, size=FONT_SIZE_TITLE),
        legend=legend_2x2(),
        width=700,
        height=640,
        **common_layout(margin=dict(t=60, b=100, l=80, r=40)),
    )
    for annotation in fig["layout"]["annotations"]:
        annotation["font"] = dict(family=FONT_FAMILY, size=FONT_SIZE_ANNOTATION)
    fig.update_yaxes(**academic_yaxis(title_text="Temperature (°C)"))
    fig.update_xaxes(**academic_xaxis(title_text="Hour of day"))
    return fig


# ── plot 6 – OL vs CO temperature with deadbands & occupancy (2×2) ───────────


def plot_temperature_comparison(inner1, inner2, buildings, labels):
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[f"{sl} – open loop vs optimised" for sl in labels],
        vertical_spacing=0.22,
        horizontal_spacing=0.18,
    )

    for idx, (b, sl) in enumerate(zip(buildings, labels)):
        row, col = divmod(idx, 2)
        row += 1
        col += 1

        temp_ol = schedule(inner1, b, "indoor_temperature")
        temp_co = schedule(inner2, b, "indoor_temperature")
        setpt = schedule(inner2, b, "temperature_setpoint")
        lower = schedule(inner2, b, "temperature_lower_deadband_celsius")
        upper = schedule(inner2, b, "temperature_upper_deadband_celsius")
        occ = schedule(inner2, b, "occupancy_profile")

        y_min = min(lower) - 1
        y_max = max(upper) + 1

        for shape in occupancy_shapes(occ, y_min, y_max):
            shape.update(xref=f"x{idx+1}", yref=f"y{idx+1}")
            fig.add_shape(shape)

        show = idx == 0

        fig.add_trace(
            go.Scatter(
                name="Upper deadband",
                x=HOURS, y=upper,
                mode="lines",
                line=dict(color=DB_LINE, width=1, dash="dot"),
                legendgroup="db_upper",
                showlegend=show,
            ),
            row=row, col=col,
        )
        fig.add_trace(
            go.Scatter(
                name="Deadband range",
                x=HOURS, y=lower,
                mode="lines",
                line=dict(color=DB_LINE, width=1, dash="dot"),
                fill="tonexty",
                fillcolor=DB_FILL,
                legendgroup="db_lower",
                showlegend=show,
            ),
            row=row, col=col,
        )

        fig.add_trace(
            go.Scatter(
                name="Setpoint",
                x=HOURS, y=setpt,
                mode="lines",
                line=dict(color=SP_COLOR, width=1.5, dash="dash"),
                legendgroup="setpt",
                showlegend=show,
            ),
            row=row, col=col,
        )

        viol_ol = [t < lo or t > hi for t, lo, hi in zip(temp_ol, lower, upper)]
        fig.add_trace(
            go.Scatter(
                name="Open loop",
                x=HOURS, y=temp_ol,
                mode="lines+markers",
                line=dict(color=OL_COLOR, width=1.8),
                marker=dict(
                    size=[5 if v else 3 for v in viol_ol],
                    color=[VIOL_COLOR if v else OL_COLOR for v in viol_ol],
                ),
                legendgroup="ol_temp",
                showlegend=show,
            ),
            row=row, col=col,
        )

        viol_co = [t < lo or t > hi for t, lo, hi in zip(temp_co, lower, upper)]
        fig.add_trace(
            go.Scatter(
                name="Central optimisation",
                x=HOURS, y=temp_co,
                mode="lines+markers",
                line=dict(color=CO_COLOR, width=1.8),
                marker=dict(
                    size=[5 if v else 3 for v in viol_co],
                    color=[VIOL_COLOR if v else CO_COLOR for v in viol_co],
                ),
                legendgroup="co_temp",
                showlegend=show,
            ),
            row=row, col=col,
        )

    fig.update_layout(
        title_text="Open loop vs central optimisation – temperature with deadband and occupancy",
        title_x=0.5,
        title_xanchor="center",
        title_font=dict(family=FONT_FAMILY, size=FONT_SIZE_TITLE),
        legend=legend_2x2(),
        width=700,
        height=640,
        **common_layout(margin=dict(t=60, b=100, l=80, r=40)),
    )
    for annotation in fig["layout"]["annotations"]:
        annotation["font"] = dict(family=FONT_FAMILY, size=FONT_SIZE_ANNOTATION)
    fig.update_yaxes(**academic_yaxis(title_text="Temperature (°C)"))
    fig.update_xaxes(**academic_xaxis(title_text="Hour of day"))
    return fig


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ol",
        default="Results/schedules/open_loop_all_buildings_peak_demand.json",
        help="Path to open-loop JSON file",
    )
    parser.add_argument(
        "--co",
        default="Results/schedules/central_optimisation_all_buildings_peak_demand.json",
        help="Path to central-optimisation JSON file",
    )
    parser.add_argument("--out", default=".", help="Output directory for saved PDFs")
    parser.add_argument("--no-show", action="store_true", help="Skip fig.show() – useful in headless environments")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.ol) as f:
        ol_data = json.load(f)
    with open(args.co) as f:
        co_data = json.load(f)

    inner1 = ol_data["all_buildings_peak_demand"]
    inner2 = co_data["all_buildings_peak_demand"]

    buildings = list(inner2.keys())
    labels = [short(b) for b in buildings]

    plots = [
        ("cost_comparison", plot_cost(inner1, inner2, buildings, labels)),
        ("thermal_deviation_comparison", plot_temp_deviation(inner1, inner2, buildings, labels)),
        ("peak_demand_summary", plot_peak_summary(inner1, inner2, buildings, labels)),
        ("grid_import_hourly", plot_grid_import(inner1, inner2, buildings, labels)),
        ("temperature_co_deadband", plot_temperature_co(inner2, buildings, labels)),
        ("temperature_ol_vs_co", plot_temperature_comparison(inner1, inner2, buildings, labels)),
        ("gas_import_comparison", plot_gas_import(inner1, inner2, buildings, labels)),
    ]

    for name, fig in plots:
        path = out_dir / f"{name}.pdf"
        fig.write_image(str(path))
        print(f"  saved → {path}")
        if not args.no_show:
            fig.show()

    print(f"\nDone – {len(plots)} plots saved to {out_dir.resolve()}")


if __name__ == "__main__":
    main()