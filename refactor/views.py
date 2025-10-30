"""Methods to generate views of data."""
from pathlib import Path
from typing import TypedDict, Callable
from enum import StrEnum

import polars as pl
import panel as pn
from panel.viewable import Viewer
import plotly.graph_objects as go
from plotly.basedatatypes import BaseTraceType
import colorcet as cc

from modules.nwm import ModelConfiguration
from modules.evaluate import load_metrics, Metric
from modules.routelink import download_routelink
from modules.pairs import GROUP_SPECIFICATIONS

pn.extension('tabulator')

class ModelDomain(StrEnum):
    """Symbols for model domains."""
    CONUS = "[CONUS] "
    ALASKA = "[Alaska] "
    HAWAII = "[Hawaii] "
    PUERTO_RICO = "[Puerto Rico] "

DOMAIN_LOOKUP: dict[ModelConfiguration, ModelDomain] = {
    ModelConfiguration.ANALYSIS_ASSIM_EXTEND_NO_DA: ModelDomain.CONUS,
    ModelConfiguration.MEDIUM_RANGE_MEM_1: ModelDomain.CONUS,
    ModelConfiguration.MEDIUM_RANGE_BLEND: ModelDomain.CONUS,
    ModelConfiguration.MEDIUM_RANGE_NO_DA: ModelDomain.CONUS,
    ModelConfiguration.SHORT_RANGE: ModelDomain.CONUS,
    ModelConfiguration.ANALYSIS_ASSIM_EXTEND_ALASKA_NO_DA: ModelDomain.ALASKA,
    ModelConfiguration.MEDIUM_RANGE_ALASKA_MEM_1: ModelDomain.ALASKA,
    ModelConfiguration.MEDIUM_RANGE_BLEND_ALASKA: ModelDomain.ALASKA,
    ModelConfiguration.MEDIUM_RANGE_ALASKA_NO_DA: ModelDomain.ALASKA,
    ModelConfiguration.SHORT_RANGE_ALASKA: ModelDomain.ALASKA,
    ModelConfiguration.ANALYSIS_ASSIM_HAWAII_NO_DA: ModelDomain.HAWAII,
    ModelConfiguration.SHORT_RANGE_HAWAII: ModelDomain.HAWAII,
    ModelConfiguration.SHORT_RANGE_HAWAII_NO_DA: ModelDomain.HAWAII,
    ModelConfiguration.ANALYSIS_ASSIM_PUERTO_RICO_NO_DA: ModelDomain.PUERTO_RICO,
    ModelConfiguration.SHORT_RANGE_PUERTO_RICO: ModelDomain.PUERTO_RICO,
    ModelConfiguration.SHORT_RANGE_PUERTO_RICO_NO_DA: ModelDomain.PUERTO_RICO
}
"""Mapping from ModelConfiguration to ModelDomain."""

CONFIGURATION_LOOKUP: dict[ModelConfiguration, str] = {
    ModelConfiguration.ANALYSIS_ASSIM_EXTEND_NO_DA: "Extended Analysis & Assimilation"
                                                    " (MRMS/Stage IV, No-DA)",
    ModelConfiguration.MEDIUM_RANGE_MEM_1: "Medium Range Deterministic (GFS)",
    ModelConfiguration.MEDIUM_RANGE_BLEND: "Medium Range Deterministic (NBM)",
    ModelConfiguration.MEDIUM_RANGE_NO_DA: "Medium Range Deterministic (GFS, No-DA)",
    ModelConfiguration.SHORT_RANGE: "Short Range (HRRR)",
    ModelConfiguration.ANALYSIS_ASSIM_EXTEND_ALASKA_NO_DA: "Extended Analysis & Assimilation"
                                                           " (MRMS/Stage IV, No-DA)",
    ModelConfiguration.MEDIUM_RANGE_ALASKA_MEM_1: "Medium Range Deterministic (GFS)",
    ModelConfiguration.MEDIUM_RANGE_BLEND_ALASKA: "Medium Range Deterministic (NBM)",
    ModelConfiguration.MEDIUM_RANGE_ALASKA_NO_DA: "Medium Range Deterministic (GFS, No-DA)",
    ModelConfiguration.SHORT_RANGE_ALASKA: "Short Range (HRRR)",
    ModelConfiguration.ANALYSIS_ASSIM_HAWAII_NO_DA: "Analysis & Assimilation (MRMS, No-DA)",
    ModelConfiguration.SHORT_RANGE_HAWAII: "Short Range (WRF-ARW)",
    ModelConfiguration.SHORT_RANGE_HAWAII_NO_DA: "Short Range (WRF-ARW, No-DA)",
    ModelConfiguration.ANALYSIS_ASSIM_PUERTO_RICO_NO_DA: "Analysis & Assimilation (MRMS, No-DA)",
    ModelConfiguration.SHORT_RANGE_PUERTO_RICO: "Short Range (WRF-ARW)",
    ModelConfiguration.SHORT_RANGE_PUERTO_RICO_NO_DA: "Short Range (WRF-ARW, No-DA)"
}
"""Mapping from ModelConfiguration to pretty strings."""

METRIC_LOOKUP: dict[str, Metric] = {
    "Kling-Gupta efficiency": Metric.KLING_GUPTA_EFFICIENCY,
    "Nash-Sutcliffe efficiency": Metric.NASH_SUTCLIFFE_EFFICIENCY,
    "Relative mean": Metric.RELATIVE_MEAN,
    "Relative standard deviation": Metric.RELATIVE_STANDARD_DEVIATION,
    "Pearson correlation coefficient": Metric.PEARSON_CORRELATION_COEFFICIENT,
    "Relative mean bias": Metric.RELATIVE_MEAN_BIAS,
    "Relative median": Metric.RELATIVE_MEDIAN,
    "Relative minimum": Metric.RELATIVE_MINIMUM,
    "Relative maximum": Metric.RELATIVE_MAXIMUM
}
"""Mapping from pretty strings to evaluation Metric."""

class FilterWidgets(Viewer):
    """Holds various data filtering widgets and values."""
    def __init__(self, **params):
        super().__init__(**params)

        # Merge domain and configuration look-ups
        self._configuration_lookup = {
            DOMAIN_LOOKUP[k]+v: (DOMAIN_LOOKUP[k], k) for k, v in CONFIGURATION_LOOKUP.items()
            }

        # Setup widgets
        self._widgets: dict[str, pn.widgets.Widget] = {
            "label": pn.widgets.Select(
                name="Evaluation",
                options=["FY2024Q1", "FY2024Q2"]
            ),
            "configuration": pn.widgets.Select(
                name="Model Configuration",
                options=list(self._configuration_lookup.keys())
            ),
            "metric": pn.widgets.Select(
                name="Metric",
                options=list(METRIC_LOOKUP.keys())
            ),
            "rank": pn.widgets.RadioBoxGroup(
                name="Flow aggregation",
                inline=True,
                options=["min", "median", "max"]
            ),
            "lead_time": pn.pane.Placeholder()
        }

        # Generate lead times
        self._lead_time_lookup: dict[ModelConfiguration, list[int]] = {}
        for c, s in GROUP_SPECIFICATIONS.items():
            self._lead_time_lookup[c] = list(
                range(0, s.lead_time_hours_max+s.window_interval, s.window_interval)
            )

        # Add lead time widget
        self._widgets["lead_time"].object = pn.widgets.DiscretePlayer(
            name="Minimum lead time (hours)",
            options=self._lead_time_lookup[self.configuration],
            show_loop_controls=False,
            visible_buttons=["previous", "next"],
            width=300,
            value=0
            )

        # Keep track of callbacks
        self._callbacks: list[Callable] = []

        # Update lead time
        def update_lead_times(event) -> None:
            # Ignore non-events
            if event is None:
                return

            # Maintain value
            if self.lead_time in self._lead_time_lookup[self.configuration]:
                value = self.lead_time
            else:
                value = 0

            # Create new widget to force refresh
            self._widgets["lead_time"].object = pn.widgets.DiscretePlayer(
                name="Minimum lead time (hours)",
                options=self._lead_time_lookup[self.configuration],
                show_loop_controls=False,
                visible_buttons=["previous", "next"],
                width=300,
                value=value
                )

            # Bind callbacks
            for c in self._callbacks:
                pn.bind(c, self._widgets["lead_time"].object.param.value, watch=True)
        pn.bind(update_lead_times, self._widgets["configuration"].param.value, watch=True)

        # Create layout
        self._layout = pn.Card(
            pn.Column(*list(self._widgets.values())),
            title="Filters",
            collapsible=False
            )

    def __panel__(self):
        return self._layout

    def bind(self, function) -> None:
        """Bind a function to all filtering widgets."""
        self._callbacks.append(function)
        for v in self._widgets.values():
            if isinstance(v, pn.pane.Placeholder):
                pn.bind(function, v.object.param.value, watch=True)
            else:
                pn.bind(function, v.param.value, watch=True)

    @property
    def label(self) -> str:
        """Currently selected evaluation label."""
        return self._widgets["label"].value

    @property
    def configuration(self) -> ModelConfiguration:
        """Currently selected model configuration."""
        return self._configuration_lookup[self._widgets["configuration"].value][1]

    @property
    def metric(self) -> Metric:
        """Currently selected evaluation metric."""
        return METRIC_LOOKUP[self._widgets["metric"].value]

    @property
    def rank(self) -> str:
        """Currently selected streamflow aggregation method."""
        return self._widgets["rank"].value

    @property
    def column(self) -> str:
        """Construct a metrics dataframe column label from current selection."""
        return self.metric + "_" + self.rank + "_point"

    @property
    def domain(self) -> ModelDomain:
        """Currently selected domain."""
        return self._configuration_lookup[self._widgets["configuration"].value][0]

    @property
    def lead_time(self) -> int:
        """Currently selected minimum lead time in hours."""
        return self._widgets["lead_time"].object.value

class TableView(Viewer):
    """Handles tabular view of data."""
    def __init__(self, **params):
        super().__init__(**params)

        # Setup placeholder
        self._table = pn.pane.Placeholder()

        # Create layout
        self._layout = self._table

    def __panel__(self):
        return self._layout

    def update(self, dataframe: pl.DataFrame) -> None:
        """Update data underlying tabular representation."""
        self._table.object = pn.widgets.Tabulator(
            dataframe.to_pandas(),
            height=500,
            show_index=False
        )

class PlotlyFigure(TypedDict):
    """
    Specifies plotly figure dict for use with panel.

    Attributes
    ----
    data: list[plotly.basedatatypes.BaseTraceType]
        List of plotly traces.
    layout: plotly.graph_objects.Layout
        Plotly layout.
    """
    data: list[BaseTraceType]
    layout: go.Layout

DEFAULT_ZOOM: dict[ModelDomain, int] = {
    ModelDomain.ALASKA: 5,
    ModelDomain.CONUS: 3,
    ModelDomain.HAWAII: 6,
    ModelDomain.PUERTO_RICO: 8
}
"""Default map zoom for each domain."""

DEFAULT_CENTER: dict[ModelDomain, dict[str, float]] = {
    ModelDomain.ALASKA: {"lat": 60.84683, "lon": -149.05659},
    ModelDomain.CONUS: {"lat": 38.83348, "lon": -93.97612},
    ModelDomain.HAWAII: {"lat": 21.24988, "lon": -157.59606},
    ModelDomain.PUERTO_RICO: {"lat": 18.21807, "lon": -66.32802}
}
"""Default map center for each domain."""

METRIC_PLOTTING_LIMITS: dict[Metric, tuple[float, float]] = {
    Metric.RELATIVE_MEAN_BIAS: (-1.0, 1.0),
    Metric.PEARSON_CORRELATION_COEFFICIENT: (-1.0, 1.0),
    Metric.NASH_SUTCLIFFE_EFFICIENCY: (-1.0, 1.0),
    Metric.RELATIVE_MEAN: (0.0, 2.0),
    Metric.RELATIVE_STANDARD_DEVIATION: (0.0, 2.0),
    Metric.RELATIVE_MEDIAN: (0.0, 2.0),
    Metric.RELATIVE_MINIMUM: (0.0, 2.0),
    Metric.RELATIVE_MAXIMUM: (0.0, 2.0),
    Metric.KLING_GUPTA_EFFICIENCY: (-1.0, 1.0)
}
"""Mapping from Metrics to plotting limits (cmin, cmax)."""

class MapView(Viewer):
    """Display data on a map."""
    def __init__(self, **params):
        super().__init__(**params)

        # Setup
        self._figure = PlotlyFigure(
            data=[go.Scattermap(
                marker=dict(
                    colorbar=dict(
                        title=dict(
                            side="right"
                            )
                        ),
                    size=15,
                    colorscale=cc.gouldian
                ),
                showlegend=False,
                name="",
                mode="markers"
            )],
            layout=go.Layout(
                showlegend=False,
                height=540,
                width=850,
                margin=dict(l=0, r=0, t=0, b=0),
                map=dict(
                    style="satellite-streets"
                ),
                clickmode="event",
                modebar=dict(
                    remove=["lasso", "select", "resetview"],
                    orientation="v"
                ),
                dragmode="zoom"
            )
        )

        # Create layout
        self._pane = pn.pane.Plotly(self._figure)

    def __panel__(self):
        return self._pane

    def update(
            self,
            dataframe: pl.DataFrame,
            column: str,
            domain: ModelDomain,
            cmin: float,
            cmax: float
            ) -> None:
        """Update map."""
        # Update markers
        self._figure["data"][0]["marker"].update(
            color=dataframe[column],
            cmin=cmin,
            cmax=cmax
        )
        self._figure["data"][0].update(
            lat=dataframe["latitude"],
            lon=dataframe["longitude"]
        )

        # Update focus
        self._figure["layout"].update(
            uirevision=domain
        )
        self._figure["layout"]["map"].update(
            center=DEFAULT_CENTER[domain],
            zoom=DEFAULT_ZOOM[domain]
        )

        # Send updates to frontend
        self._pane.object = self._figure

def main() -> None:
    """Main."""
    root = Path("/ised/nwm_explorer_data")

    routelink = download_routelink(root).select(
        ["nwm_feature_id", "latitude", "longitude"]
    ).collect()

    filter_widgets = FilterWidgets()

    site_map = MapView()

    def handle_widget_updates(event: str) -> None:
        # Ignore non-calls
        if event is None:
            return

        # Retrieve data
        data = load_metrics(
            root=root,
            label=filter_widgets.label,
            configuration=filter_widgets.configuration,
            metric=filter_widgets.metric,
            lead_time_hours_min=filter_widgets.lead_time,
            rank=filter_widgets.rank,
            cache=True
        ).with_columns(
            latitude=pl.col("nwm_feature_id").replace_strict(
                old=routelink["nwm_feature_id"].implode(),
                new=routelink["latitude"].implode()
            ),
            longitude=pl.col("nwm_feature_id").replace_strict(
                old=routelink["nwm_feature_id"].implode(),
                new=routelink["longitude"].implode()
            )
        )

        # Update table
        site_map.update(
            dataframe=data,
            column=filter_widgets.column,
            domain=filter_widgets.domain,
            cmin=METRIC_PLOTTING_LIMITS[filter_widgets.metric][0],
            cmax=METRIC_PLOTTING_LIMITS[filter_widgets.metric][1]
            )
    handle_widget_updates(filter_widgets.label)
    filter_widgets.bind(handle_widget_updates)

    pn.serve(pn.Row(filter_widgets, site_map))

if __name__ == "__main__":
    main()
