"""Generate and plot histograms."""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
import polars as pl
import panel as pn

from nwm_explorer.data.mapping import Metric
from nwm_explorer.interfaces.filters import METRIC_STRINGS, CONFIDENCE_STRINGS
from nwm_explorer.plots.site_map import METRIC_PLOTTING_LIMITS

import plotly.graph_objects as go

METRIC_STRING_LOOKUP: dict[Metric, str] = {v: k for k, v in METRIC_STRINGS.items()}
"""Reverse look-up from metric column slugs to pretty strings."""

class Histogram:
    def __init__(self, columns: list[Metric]):
        self.columns = columns
        self.data = {c: [go.Bar(showlegend=False, name="")] for c in columns}
        self.layouts = {k: go.Layout(
            dragmode=False,
            showlegend=False,
            height=250,
            width=300,
            margin=dict(l=0, r=0, t=0, b=0),
            yaxis=dict(title=dict(text="Frequency (%)"), range=[0.0, 100.0]),
            xaxis=dict(title=dict(text=METRIC_STRING_LOOKUP[k]))) for k in self.data}
        self.figures = {k: {"data": self.data[k], "layout": self.layouts[k]} for k in self.data}
        self.plots = {k: pn.pane.Plotly(f, config={"displayModeBar": False}) for k, f in self.figures.items()}
        
        # Bins
        self.bins = {}
        self.abscissa = {}
        self.custom_data = {}
        for k in self.data:
            cmin, cmax = METRIC_PLOTTING_LIMITS[k]
            self.bins[k] = np.linspace(cmin, cmax, 11)
            windows = sliding_window_view(self.bins[k], window_shape=2)
            abscissa = windows.mean(axis=1)
            abscissa = np.insert(abscissa, 0, abscissa[0] - (abscissa[1] - abscissa[0]))
            abscissa = np.append(abscissa, abscissa[-1] + (abscissa[-1] - abscissa[-2]))
            self.abscissa[k] = abscissa

            # Bin range
            bin_ranges = [f"<{cmin:.1f}"]
            for idx in range(len(self.bins[k])-1):
                left = self.bins[k][idx]
                right = self.bins[k][idx+1]
                bin_ranges.append(f"{left:.1f} to {right:.1f}")
            bin_ranges.append(f">{cmax:.1f}")

            # Custom data
            self.custom_data[k] = pd.DataFrame({
                "counts": np.zeros(len(bin_ranges)),
                "bin_range": bin_ranges,
                "counts_low": np.zeros(len(bin_ranges)),
                "counts_high": np.zeros(len(bin_ranges)),
                "lower_estimate": np.zeros(len(bin_ranges)),
                "upper_estimate": np.zeros(len(bin_ranges))
            })

            # Update plot data
            self.data[k][0].update(
                x=self.abscissa[k]
            )
    
    def update(self, data: pl.DataFrame) -> None:
        for k in self.data:
            count = {}
            cmin, cmax = METRIC_PLOTTING_LIMITS[k]
            for conf in CONFIDENCE_STRINGS.values():
                # Count sites
                a = data[k+conf].to_numpy()
                number_of_sites, _ = np.histogram(a, bins=self.bins[k], density=False)

                # Outside range
                number_low = a[a < cmin].size
                number_high = a[a > cmax].size

                # Expand original count
                number_of_sites = np.insert(number_of_sites, 0, number_low)
                number_of_sites = np.append(number_of_sites, number_high)

                count[conf] = number_of_sites

            # Determine histogram error bars
            estimates = np.vstack(tuple(count.values()))
            e_lo = count["_point"] - np.min(estimates, axis=0)
            e_hi = np.max(estimates, axis=0) - count["_point"]

            # Custom data
            total_sites = np.sum(count["_point"])
            self.custom_data[k]["counts"] = count["_point"]
            self.custom_data[k]["counts_low"] = np.min(estimates, axis=0)
            self.custom_data[k]["counts_high"] = np.max(estimates, axis=0)
            self.custom_data[k]["lower_estimate"] = 100 * np.min(estimates, axis=0) / total_sites
            self.custom_data[k]["upper_estimate"] = 100 * np.max(estimates, axis=0) / total_sites

            # Update plot data
            self.data[k][0].update(
                y=100 * count["_point"] / total_sites,
                customdata=self.custom_data[k],
                hovertemplate=(
                f"{METRIC_STRING_LOOKUP[k]}<br>"
                "Bin range: %{customdata[1]} <br>"
                "Sites: %{customdata[0]} of "
                f"{total_sites} "
                "(%{y:.1f} %)<br>"
                "95% CI: %{customdata[2]} to %{customdata[3]} sites<br>"
                "        (%{customdata[4]:.1f} to %{customdata[5]:.1f} %)"
                ),
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=100 * e_hi / total_sites,
                    arrayminus=100 * e_lo / total_sites
                    )
            )

    def refresh(self) -> None:
        self.figures = {k: {"data": self.data[k], "layout": self.layouts[k]} for k in self.data}
        for k in self.plots:
            self.plots[k].object = self.figures[k]

    def servable(self) -> pn.GridBox:
        cards = [pn.Card(
            v,
            collapsible=False,
            hide_header=True
        ) for v in self.plots.values()]
        return pn.GridBox(*cards, ncols=2)
