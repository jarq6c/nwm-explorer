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
import pandas as pd
import numpy.typing as npt

from modules.nwm import ModelConfiguration, nwm_site_generator
from modules.evaluate import load_metrics, Metric, scan_evaluations
from modules.routelink import download_routelink
from modules.pairs import GROUP_SPECIFICATIONS, NWMGroupSpecification
from modules.usgs import usgs_site_generator

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

RANK_LOOKUP: dict[str, str] = {
    "Median": "median",
    "Minimum": "min",
    "Maximum": "max"
}
"""Mapping from pretty strings to column label components. 'Rank' refers to the
aggregation method used to resample streamflow.
"""

def build_lead_time_lookup(
        group_specificiations: dict[ModelConfiguration, NWMGroupSpecification] | None = None
) -> dict[ModelConfiguration, list[int]]:
    """build."""
    # Check specifications
    if group_specificiations is None:
        group_specificiations = GROUP_SPECIFICATIONS

    # Build lead time lists
    lead_time_lookup = {}
    for c, s in group_specificiations.items():
        lead_time_lookup[c] = list(
            range(0, s.lead_time_hours_max+s.window_interval, s.window_interval)
        )
    return lead_time_lookup

def build_lead_time_strings(
        lead_time_lookup: dict[ModelConfiguration, list[int]] | None = None
) -> dict[ModelConfiguration, list[str]]:
    """build."""
    # Check specifications
    if lead_time_lookup is None:
        lead_time_lookup = build_lead_time_lookup()

    # Build lead time lists
    lead_time_strings = {}
    for c, lead_times in lead_time_lookup.items():
        if len(lead_times) == 1:
            lead_time_strings[c] = [str(l) for l in lead_times]
            continue

        lead_time_strings[c] = []
        for idx in range(0, len(lead_times)-1):
            lead_time_strings[c].append(f"{lead_times[idx]} to <{lead_times[idx+1]}")
        lead_time_strings[c].append(f"{lead_times[-1]}")

    return lead_time_strings

class FilterWidgets(Viewer):
    """Holds various data filtering widgets and values."""
    def __init__(self, evaluation_options: list[str], **params):
        super().__init__(**params)

        # Merge domain and configuration look-ups
        self._configuration_lookup = {
            DOMAIN_LOOKUP[k]+v: (DOMAIN_LOOKUP[k], k) for k, v in CONFIGURATION_LOOKUP.items()
            }

        # Setup widgets
        self._widgets: dict[str, pn.widgets.Widget] = {
            "label": pn.widgets.Select(
                name="Evaluation",
                options=evaluation_options
            ),
            "configuration": pn.widgets.Select(
                name="Model Configuration",
                options=list(self._configuration_lookup.keys())
            ),
            "metric": pn.widgets.Select(
                name="Metric",
                options=list(METRIC_LOOKUP.keys())
            ),
            "rank": pn.widgets.Select(
                name="Streamflow aggregation method",
                options=list(RANK_LOOKUP.keys())
            ),
            "significant": pn.widgets.Select(
                name="Show",
                options=["All sites", "Statistically significant"]
            ),
            "lead_time": pn.pane.Placeholder()
        }

        # Generate lead times
        self._lead_time_lookup = build_lead_time_lookup()
        self._lead_time_strings = build_lead_time_strings()

        # Add lead time widget
        self._widgets["lead_time"].object = pn.widgets.DiscretePlayer(
            name="Lead time (hours)",
            options=self._lead_time_strings[self.configuration],
            show_loop_controls=False,
            visible_buttons=["previous", "next"],
            width=300,
            value=self._lead_time_strings[self.configuration][0]
            )

        # Keep track of callbacks
        self._callbacks: list[Callable] = []

        # Update lead time
        def update_lead_times(event) -> None:
            # Ignore non-events
            if event is None:
                return

            # Maintain value
            value = self._widgets["lead_time"].object.value
            if value not in self._lead_time_strings[self.configuration]:
                value = self._lead_time_strings[self.configuration][0]

            # Create new widget to force refresh
            self._widgets["lead_time"].object = pn.widgets.DiscretePlayer(
                name="Lead time (hours)",
                options=self._lead_time_strings[self.configuration],
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
    def metric_label(self) -> str:
        """Currently selected evaluation metric label."""
        return self._widgets["metric"].value

    @property
    def rank(self) -> str:
        """Currently selected streamflow aggregation method."""
        return RANK_LOOKUP[self._widgets["rank"].value]

    @property
    def significant(self) -> bool:
        """Only show 'statistically significant sites."""
        return self._widgets["significant"].value == "Statistically significant"

    @property
    def point_column(self) -> str:
        """Construct a metrics dataframe column label from current selection."""
        return self.metric + "_" + self.rank + "_point"

    @property
    def upper_column(self) -> str:
        """Construct a metrics dataframe column label from current selection."""
        return self.metric + "_" + self.rank + "_upper"

    @property
    def lower_column(self) -> str:
        """Construct a metrics dataframe column label from current selection."""
        return self.metric + "_" + self.rank + "_lower"

    @property
    def domain(self) -> ModelDomain:
        """Currently selected domain."""
        return self._configuration_lookup[self._widgets["configuration"].value][0]

    @property
    def lead_time(self) -> int:
        """Currently selected minimum lead time in hours."""
        lead_time_string = self._widgets["lead_time"].object.value
        idx = self._lead_time_strings[self.configuration].index(lead_time_string)
        return self._lead_time_lookup[self.configuration][idx]

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
                clickmode="event+select",
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
        return pn.Card(
            self._pane,
            collapsible=False,
            hide_header=True
            )

    def bind_click(self, function) -> None:
        """Bind a function to click event."""
        pn.bind(function, self._pane.param.click_data, watch=True)

    def update(
            self,
            dataframe: pl.DataFrame,
            column: str,
            domain: ModelDomain,
            cmin: float,
            cmax: float,
            metric_label: str,
            custom_data: pd.DataFrame | None = None,
            hover_template: str | None = None
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
            lon=dataframe["longitude"],
            customdata=custom_data,
            hovertemplate=hover_template
        )
        self._figure["data"][0]["marker"]["colorbar"]["title"].update(text=metric_label)

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

class TimeSeriesView(Viewer):
    """Display time series data."""
    def __init__(self, **params):
        super().__init__(**params)

        # Setup
        self._figure = PlotlyFigure(
            data=[go.Scatter()],
            layout=go.Layout(
                height=250,
                width=1045,
                margin=dict(l=0, r=0, t=0, b=0)
            )
        )

        # Create layout
        self._pane = pn.pane.Plotly(self._figure)

    def __panel__(self):
        return pn.Card(
            self._pane,
            collapsible=False,
            hide_header=True
            )

    def erase_data(
            self,
            xrange: tuple[pd.Timestamp, pd.Timestamp] | None = None
        ) -> None:
        """Completely erase displayed time series."""
        # Overwrite old traces
        self._figure["data"] = [go.Scatter(
            mode="lines",
            line={"color": "#3C00FF", "width": 2.0}
        )]

        # Update x-range
        if xrange is not None:
            self._figure["layout"]["xaxis"].update(range=xrange)

        # Refresh
        self._pane.object = self._figure

    def update_trace(
            self,
            xdata: npt.ArrayLike,
            ydata: npt.ArrayLike,
            index: int = 0,
            name: str | None = None,
        ) -> None:
        """Update specific time series."""
        # Update x-y data
        self._figure["data"][index].update(
            x=xdata,
            y=ydata
        )

        # Update name
        if name:
            self._figure["data"][index].update(
                name=name
            )

        # Refresh
        self._pane.object = self._figure

    def append_traces(
            self,
            traces: list[tuple[npt.ArrayLike, npt.ArrayLike, str]]
        ) -> None:
        """Add time series."""
        # Add new traces
        for trace in traces:
            self._figure["data"].append(go.Scatter(
                x=trace[0],
                y=trace[1],
                mode="lines",
                line={"color": "#FF003C", "width": 1.0},
                name=trace[2]
            ))

        # Refresh
        self._pane.object = self._figure

def main() -> None:
    """Main."""
    root = Path("/ised/nwm_explorer_data")

    routelink = download_routelink(root).select(
        ["nwm_feature_id", "latitude", "longitude"]
    ).collect()

    filter_widgets = FilterWidgets(
        evaluation_options=scan_evaluations(root, cache=True).select(
            "label").collect().unique()["label"].to_list()
    )
    data_ranges: dict[str, pd.Timestamp] = {
        "observed_value_time_min": None,
        "observed_value_time_max": None,
        "reference_time_min": None,
        "reference_time_max": None,
    }
    state: dict[str, str] = {}

    site_map = MapView()
    hydrograph = TimeSeriesView()

    def handle_filter_updates(event: str) -> None:
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
            additional_columns=(
                "nwm_feature_id",
                "usgs_site_code",
                "sample_size",
                "observed_value_time_min",
                "observed_value_time_max",
                "reference_time_min",
                "reference_time_max"
                ),
            significant=filter_widgets.significant,
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

        # Update date range
        data_ranges["observed_value_time_min"] = data["observed_value_time_min"].min()
        data_ranges["observed_value_time_max"] = data["observed_value_time_max"].max()
        data_ranges["reference_time_min"] = data["reference_time_min"].min()
        data_ranges["reference_time_max"] = data["reference_time_max"].max()

        # Update map
        site_map.update(
            dataframe=data,
            column=filter_widgets.point_column,
            domain=filter_widgets.domain,
            cmin=METRIC_PLOTTING_LIMITS[filter_widgets.metric][0],
            cmax=METRIC_PLOTTING_LIMITS[filter_widgets.metric][1],
            metric_label=filter_widgets.metric_label,
            custom_data=data.to_pandas()[[
                    "nwm_feature_id",
                    "usgs_site_code",
                    filter_widgets.lower_column,
                    filter_widgets.upper_column,
                    "sample_size"
                ]],
            hover_template=(
                    f"{filter_widgets.metric_label}: "
                    "%{marker.color:.2f}<br>"
                    "95% CI: %{customdata[2]:.2f} -- %{customdata[3]:.2f}<br>"
                    "Samples: %{customdata[4]:.0f}<br>"
                    "NWM Feature ID: %{customdata[0]}<br>"
                    "USGS Site Code: %{customdata[1]}<br>"
                    "Longitude: %{lon}<br>"
                    "Latitude: %{lat}"
                )
            )
    handle_filter_updates(filter_widgets.label)
    filter_widgets.bind(handle_filter_updates)

    def handle_click(event) -> None:
        if event is None:
            return

        # Parse custom data
        metadata = event["points"][0]["customdata"]
        nwm_feature_id = metadata[0]
        usgs_site_code = metadata[1]

        # Ignore duplicate selections
        if state.get("site") == usgs_site_code:
            return
        state["site"] = usgs_site_code

        # Stream observations
        hydrograph.erase_data(
            xrange=(
                data_ranges["observed_value_time_min"],
                data_ranges["observed_value_time_max"]
                )
        )
        dataframes = []
        for df in usgs_site_generator(
            root=root,
            usgs_site_code=usgs_site_code,
            start_time=data_ranges["observed_value_time_min"],
            end_time=data_ranges["observed_value_time_max"],
            cache=True
        ):
            # Append data
            dataframes.append(df)
            observations = pl.concat(dataframes)

            # Replace data
            hydrograph.update_trace(
                xdata=observations["value_time"].to_numpy(),
                ydata=observations["observed_cfs"].to_numpy(),
                name=f"USGS-{usgs_site_code}"
            )

        # Stream predictions
        for df in nwm_site_generator(
            root=root,
            configuration=filter_widgets.configuration,
            nwm_feature_id=nwm_feature_id,
            start_time=data_ranges["reference_time_min"],
            end_time=data_ranges["reference_time_max"],
            cache=True
        ):
            # Add each reference time
            trace_data = []
            for rt in df["reference_time"].unique():
                # Extract predictions
                predictions = df.filter(pl.col("reference_time") == rt)

                # Add trace data
                trace_data.append((
                    predictions["value_time"].to_numpy(),
                    predictions["predicted_cfs"].to_numpy(),
                    str(rt)
                ))

            # Add to plot
            hydrograph.append_traces(trace_data)
    site_map.bind_click(handle_click)

    pn.serve(pn.Column(
        pn.Row(filter_widgets, site_map),
        hydrograph)
        )

if __name__ == "__main__":
    main()
