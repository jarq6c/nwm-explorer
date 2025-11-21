"""Methods to generate views of data."""
from typing import Callable, Any, Generator
from itertools import cycle, count
from dataclasses import dataclass

import polars as pl
import panel as pn
from panel.viewable import Viewer
import plotly.graph_objects as go
from plotly.colors import hex_to_rgb
import colorcet as cc
import pandas as pd
import numpy.typing as npt
import numpy as np
import geopandas as gpd

from .nwm import ModelConfiguration
from .evaluate import Metric
from .constants import (ModelDomain, GROUP_SPECIFICATIONS, DOMAIN_LOOKUP,
    CONFIGURATION_LOOKUP, NWMGroupSpecification, METRIC_LOOKUP, RANK_LOOKUP,
    PlotlyFigure, DEFAULT_CENTER, DEFAULT_ZOOM, AxisType)
from .configuration import MapLayer

pn.extension('tabulator')

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
        def update_lead_times(event, callback_type: str) -> None:
            # Ignore non-events
            if event is None or callback_type is None:
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
                pn.bind(c, self._widgets["lead_time"].object.param.value, watch=True,
                    callback_type="lead_time")
        pn.bind(update_lead_times, self._widgets["configuration"].param.value, watch=True,
            callback_type="configuration")

        # Create layout
        self._layout = pn.Card(
            pn.Column(*list(self._widgets.values())),
            title="Filters",
            collapsible=False,
            width=350
            )

    def __panel__(self):
        return self._layout

    def bind(self, function) -> None:
        """Bind a function to all filtering widgets."""
        self._callbacks.append(function)
        for k, v in self._widgets.items():
            if isinstance(v, pn.pane.Placeholder):
                pn.bind(function, v.object.param.value, watch=True, callback_type=k)
            else:
                pn.bind(function, v.param.value, watch=True, callback_type=k)

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
        return pn.Card(
            self._layout,
            collapsible=False,
            hide_header=True
            )

    def update(self, dataframe: pl.DataFrame) -> None:
        """Update data underlying tabular representation."""
        self._table.object = pn.widgets.Tabulator(
            dataframe.to_pandas(),
            height=500,
            show_index=False
        )

class MapView(Viewer):
    """Display data on a map."""
    def __init__(self, map_layers: list[MapLayer] | None = None, **params):
        super().__init__(**params)

        # Setup extra layers
        index = count(2)
        self.layers = {l.name: (next(index), l) for l in map_layers}
        self.layer_selector = pn.widgets.CheckBoxGroup(
            name="Map Layers",
            options=[l.name for l in map_layers]
        )
        self.map_layer_selector = pn.Column(
            "# Map Layers",
            self.layer_selector
        )

        # Stub layers
        layer_color = cycle(cc.glasbey_light)
        layer_data = [
            go.Scattermap(
                    showlegend=False,
                    name="",
                    mode="markers",
                    marker=dict(size=10, color=next(layer_color))
                ) for _ in map_layers
        ]

        # Setup main map
        self._domain = None
        self._figure = PlotlyFigure(
            data=[
                go.Scattermap(
                    showlegend=False,
                    name="",
                    mode="markers",
                    marker=dict(size=20, color="#00FFF2")
                ),
                go.Scattermap(
                    marker=dict(
                        colorbar=dict(
                            title=dict(
                                side="right"
                                )
                            ),
                        size=15,
                        colorscale=cc.CET_L17
                    ),
                    showlegend=False,
                    name="",
                    mode="markers"
                    )
                ] + layer_data,
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
        self._pane = pn.pane.Plotly(self._figure, config={'displaylogo': False})

        # Track selection state
        self.click_data: dict[str, Any] = {}
        def catch_click(event, callback_type: str) -> None:
            if event is None or callback_type is None:
                return

            # Handle selection/deselection
            lat, lon = event["points"][0].get("lat"), event["points"][0].get("lon")
            if (self.click_data.get("lat") == lat) & (self.click_data.get("lon") == lon):
                self.click_data = {}
            else:
                self.click_data.update(event["points"][0])

            # Update marker
            self.update_marker()
        pn.bind(catch_click, self._pane.param.click_data, watch=True, callback_type="click")

        # Viewport state
        self.viewport: dict[str, float] = {}
        def catch_relayout(event: dict[str, Any], callback_type: str) -> None:
            if event is None or callback_type is None:
                return

            # Check for viewport change
            bbox = event.get("map._derived")
            if bbox is None:
                return

            # Extract points
            self.viewport = {
                "lat_max": bbox["coordinates"][0][1],
                "lat_min": bbox["coordinates"][2][1],
                "lon_max": bbox["coordinates"][1][0],
                "lon_min": bbox["coordinates"][0][0]
            }
        pn.bind(catch_relayout, self._pane.param.relayout_data, watch=True,
            callback_type="relayout")

        # Layer selection
        def handle_layer_selection(event: list[str], callback_type: str) -> None:
            if event is None or callback_type is None:
                return

            # Select/deselect each layer
            for name, (idx, layer) in self.layers.items():
                if name in event and self._figure["data"][idx]["lat"] is None:
                    # Load data
                    if layer.columns is None:
                        cols = ["geometry"]
                    elif "geometry" not in layer.columns:
                        cols = layer.columns+["geometry"]
                    else:
                        cols = layer.columns
                    gdf = gpd.read_parquet(layer.path, columns=cols)

                    # Custom data
                    if len(cols) != 1:
                        custom_data = gdf[layer.columns]
                    else:
                        custom_data = None

                    # Hover template
                    hover_template = ""
                    for cindex, cname in enumerate(layer.columns):
                        hover_template += f"{cname}: " + "%{customdata[" + str(cindex) + "]}<br>"
                    hover_template += "Longitude: %{lon}<br>Latitude: %{lat}"

                    # Update trace
                    self._figure["data"][idx].update(
                        lat=gdf["geometry"].y,
                        lon=gdf["geometry"].x,
                        customdata=custom_data,
                        hovertemplate=hover_template
                    )
                else:
                    self._figure["data"][idx].update(
                        lat=None,
                        lon=None,
                        customdata=None,
                        hovertemplate=None
                    )

            # Update
            self._pane.object = self._figure
        pn.bind(handle_layer_selection, self.layer_selector.param.value, watch=True,
            callback_type="layer")

    def __panel__(self):
        return pn.Card(
            self._pane,
            collapsible=False,
            hide_header=True
            )

    def update_marker(self) -> None:
        """Move or remove selection marker."""
        self._figure["data"][0].update(
            lat=[self.click_data.get("lat")],
            lon=[self.click_data.get("lon")]
        )
        self._pane.object = self._figure

    def bind_click(self, function) -> None:
        """Bind a function to click event."""
        pn.bind(function, self._pane.param.click_data, watch=True, callback_type="click")

    def bind_relayout(self, function) -> None:
        """Bind a function to relayout event."""
        pn.bind(function, self._pane.param.relayout_data, watch=True, callback_type="relayout")

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
        # Update domain
        if self._domain != domain:
            self._domain = domain
            self.viewport = {}

        # Handle selection
        show_marker = self.click_data.get("lat") in dataframe["latitude"]
        show_marker &= self.click_data.get("lon") in dataframe["longitude"]
        if not show_marker:
            self.click_data = {}
            self.update_marker()

        # Update markers
        self._figure["data"][1]["marker"].update(
            color=dataframe[column],
            cmin=cmin,
            cmax=cmax
        )
        self._figure["data"][1].update(
            lat=dataframe["latitude"],
            lon=dataframe["longitude"],
            customdata=custom_data,
            hovertemplate=hover_template
        )
        self._figure["data"][1]["marker"]["colorbar"]["title"].update(text=metric_label)

        # Update focus
        self._figure["layout"].update(
            uirevision=domain,
            selectionrevision=(self.click_data.get("lat"), self.click_data.get("lon"))
        )
        self._figure["layout"]["map"].update(
            center=DEFAULT_CENTER[domain],
            zoom=DEFAULT_ZOOM[domain]
        )

        # Send updates to frontend
        self._pane.object = self._figure

def rgb_to_hex(rgb_tuple):
    """Converts an RGB tuple to a hexadecimal color string."""
    r, g, b = rgb_tuple
    return f"#{r:02x}{g:02x}{b:02x}"

def invert_color(value: str) -> str:
    """Convert a hex color to an inverted hex color.
    
    Parameters
    ----------
    value: str, required,
        Hex color string.
    
    Returns
    -------
    str:
        Inverted hex color.
    """
    r, g, b = hex_to_rgb(value)
    return rgb_to_hex((255-r, 255-g, 255-b))

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
                margin=dict(l=0, r=25, t=10, b=0),
                yaxis=dict(title=dict(text="Streamflow (CFS)")),
                clickmode="event",
                modebar=dict(
                    remove=["resetview"],
                    orientation="v"
                ),
                showlegend=False
            )
        )

        # Create layout
        self._pane = pn.pane.Plotly(self._figure, config={'displaylogo': False})

        # Color ramp
        self._color_ramp = cycle(cc.CET_C7)

        # Track selection state
        self.click_data: dict[str, Any] = {}
        self.selected_traces: dict[int, bool] = {}
        def catch_click(event, callback_type: str) -> None:
            if event is None or callback_type is None:
                return

            # Get curve number
            curve_number = event["points"][0].get("curveNumber")

            # Highlight/unhighlight trace
            self.toggle_highlight(curve_number)

            # Update click data
            if self.click_data.get("curveNumber") == curve_number:
                self.click_data = {}
            else:
                self.click_data.update(event["points"][0])
        pn.bind(catch_click, self._pane.param.click_data, watch=True, callback_type="click")

    def __panel__(self):
        return pn.Card(
            self._pane,
            collapsible=False,
            hide_header=True
            )

    def set_axis_type(self, axis_type: AxisType) -> None:
        """Change y-axis type."""
        # Update
        self._figure["layout"]["yaxis"].update(type=axis_type)

        # Refresh
        self._pane.object = self._figure

    def toggle_highlight(self, curve_number: int) -> None:
        """Highlight or unhighlight selected trace."""
        # Invert colors
        color = self._figure["data"][curve_number]["line"]["color"]
        self._figure["data"][curve_number]["line"].update(
            color=invert_color(color)
        )

        # Update width
        width = self._figure["data"][curve_number]["line"]["width"]
        if curve_number in self.selected_traces:
            # Remove trace
            self.selected_traces.pop(curve_number)

            # Reduce width
            self._figure["data"][curve_number]["line"].update(
                width=width-4
            )
        else:
            # Add trace
            self.selected_traces[curve_number] = True

            # Increase width
            self._figure["data"][curve_number]["line"].update(
                width=width+4
            )

        # Refresh
        self._pane.object = self._figure

    def erase_data(
            self,
            xrange: tuple[pd.Timestamp, pd.Timestamp] | None = None
        ) -> None:
        """Completely erase displayed time series."""
        # Overwrite old traces
        self._figure["data"] = [go.Scatter(
            mode="lines",
            line={"color": "#000000", "width": 4.0},
            name=""
        )]

        # Update x-range
        if xrange is not None:
            self._figure["layout"]["xaxis"].update(range=xrange)

        # Update focus
        self._figure["layout"].update(
            uirevision=pd.Timestamp.now().strftime("%Y%m%dT%H%M%S")
        )

        # Refresh
        self._pane.object = self._figure

        # Clear click data
        self.click_data = {}
        self.selected_traces = {}

    def update_trace(
            self,
            xdata: npt.ArrayLike,
            ydata: npt.ArrayLike,
            index: int = 0,
            name: str | None = None,
        ) -> None:
        """Update specific time series."""
        # Y-axis label
        ylabel = self._figure["layout"]["yaxis"]["title"]["text"]

        # Update x-y data
        self._figure["data"][index].update(
            x=xdata,
            y=ydata,
            hovertemplate=(
                f"<b>{name}</b><br>"
                "Valid: %{x}<br>"
                f"{ylabel}: " + "%{y:.2f}"
                )
        )

        # Refresh
        self._pane.object = self._figure

    def append_traces(
            self,
            traces: list[tuple[npt.ArrayLike, npt.ArrayLike, str]],
            mode: str = "lines"
        ) -> None:
        """Add time series."""
        # Y-axis label
        ylabel = self._figure["layout"]["yaxis"]["title"]["text"]

        # Add new traces
        for trace in traces:
            self._figure["data"].append(go.Scatter(
                x=trace[0],
                y=trace[1],
                mode=mode,
                line={"color": next(self._color_ramp), "width": 2.0},
                name="",
                hovertemplate=(
                    f"<b>{trace[2]}</b><br>"
                    "Valid: %{x}<br>"
                    f"{ylabel}: " + "%{y:.2f}"
                    )
            ))

        # Refresh
        self._pane.object = self._figure

class BarPlot(Viewer):
    """Display barplots."""
    def __init__(self, **params) -> None:
        super().__init__(**params)

        # Setup
        self.data = [go.Bar(
            name=""
        )]
        self.layout = go.Layout(
            height=250,
            width=440,
            margin=dict(l=0, r=25, t=10, b=0),
            modebar=dict(
                remove=["lasso", "select", "pan", "autoscale", "zoomin", "zoomout"],
                orientation="v"
            )
        )
        self.figure = dict(data=self.data, layout=self.layout)
        self.pane = pn.pane.Plotly(self.figure, config={'displaylogo': False})

    def update(
            self,
            xdata: npt.ArrayLike,
            ydata: npt.ArrayLike,
            ydata_lower: npt.ArrayLike,
            ydata_upper: npt.ArrayLike,
            xlabel: str,
            ylabel: str
        ) -> None:
        """Update data."""
        # Construct custom data
        custom_data = np.hstack((ydata_lower[:, np.newaxis], ydata_upper[:, np.newaxis]))

        # Update trace
        self.data[0].update(dict(
            x=xdata,
            y=ydata,
            customdata=custom_data,
            hovertemplate=(
                f"{ylabel}: " + "%{y:.2f}<br>" +
                "95% CI: %{customdata[0]:.2f} -- %{customdata[1]:.2f}<br>" +
                f"{xlabel}: " + "%{x}"
            ),
            error_y=dict(
                type="data",
                array=ydata_upper - ydata,
                arrayminus=ydata - ydata_lower
            )
        ))

        # Update axes
        self.layout.update(dict(
            xaxis=dict(title=dict(text=xlabel)),
            yaxis=dict(title=dict(text=ylabel))
        ))

        # Update frontend
        self.figure.update(dict(data=self.data, layout=self.layout))
        self.pane.object = self.figure

    def erase(
            self
        ) -> None:
        """Erase data."""
        self.data = [go.Bar(
            name=""
        )]

        # Update frontend
        self.figure.update(dict(data=self.data, layout=self.layout))
        self.pane.object = self.figure

    def __panel__(self) -> pn.Card:
        return pn.Card(
            self.pane,
            collapsible=False,
            hide_header=True
        )

class ECDFPlot(Viewer):
    """Display empirical cumulative distribution curves."""
    def __init__(self, **params) -> None:
        super().__init__(**params)

        # Setup
        self.data = [go.Scatter(
            name="",
            mode="markers"
        )]
        self.layout = go.Layout(
            height=255,
            width=300,
            margin=dict(l=0, r=25, t=10, b=0),
            modebar=dict(
                remove=["lasso", "select"],
                orientation="v"
            )
        )
        self.figure = dict(data=self.data, layout=self.layout)
        self.pane = pn.pane.Plotly(self.figure, config={'displaylogo': False})

    def update(
            self,
            xdata: npt.ArrayLike,
            xdata_lower: npt.ArrayLike,
            xdata_upper: npt.ArrayLike,
            xlabel: str,
            ylabel: str = "Empirical CDF",
            xrange: tuple[float, float] | None = None
        ) -> None:
        """Update data. Assumes xdata are sorted."""
        # Set xrange
        if xrange is None:
            xrange = (-2, 2)

        # Construct custom data
        custom_data = np.hstack((xdata_lower[:, np.newaxis], xdata_upper[:, np.newaxis]))

        # Plotting position
        ydata = np.arange(len(xdata)) / (len(xdata) + 1)

        # Update trace
        self.data[0].update(dict(
            x=xdata,
            y=ydata,
            customdata=custom_data,
            hovertemplate=(
                f"{xlabel}: " + "%{x:.2f}<br>" +
                "95% CI: %{customdata[0]:.2f} to %{customdata[1]:.2f}<br>" +
                f"{ylabel}: " + "%{y:.2f}"
            )
        ))

        # Update axes
        self.layout.update(dict(
            xaxis=dict(
                title=dict(text=xlabel),
                range=(xrange[0], xrange[1]),
                tickmode="linear",
                tick0=xrange[0],
                dtick=(xrange[1]-xrange[0])/4,
                tickformat=".1f"
            ),
            yaxis=dict(
                title=dict(text=ylabel),
                range=(0, 1),
                tickmode="linear",
                tick0=0.0,
                dtick=0.2,
                tickformat=".1f"
            )
        ))

        # Update frontend
        self.figure.update(dict(data=self.data, layout=self.layout))
        self.pane.object = self.figure

    def erase(
            self
        ) -> None:
        """Erase data."""
        self.data = [go.Scatter(
            name="",
            mode="markers"
        )]

        # Update frontend
        self.figure.update(dict(data=self.data, layout=self.layout))
        self.pane.object = self.figure

    def __panel__(self) -> pn.Card:
        return pn.Card(
            self.pane,
            collapsible=False,
            hide_header=True
        )

class ECDFMatrix(Viewer):
    """Show a set of linked empirical CDF plots."""
    def __init__(self, nplots: int, ncols: int, **params):
        super().__init__(**params)

        # Setup
        self.ncols = ncols
        self.plots = [ECDFPlot() for _ in range(nplots)]

    def update(self, index: int, **kwargs) -> None:
        """Update a plot."""
        self.plots[index].update(**kwargs)

    def erase(self, index: int) -> None:
        """Erase a single plot."""
        self.plots[index].erase()

    def erase_all(self) -> None:
        """Erase all plots."""
        for p in self.plots:
            p.erase()

    def __panel__(self) -> pn.GridBox:
        return pn.GridBox(*self.plots, ncols=self.ncols)

@dataclass
class ECDFParameters:
    """Dataclass used to hold empirical CDF plotting parameters."""
    index: int
    metric: Metric
    metric_label: str
    point_column: str
    upper_column: str
    lower_column: str

class ECDFSelector(Viewer):
    """WidetBox with metric selectors corresponding to ECDF plots."""
    def __init__(self, nplots: int, filter_widgets: FilterWidgets, **params):
        super().__init__(**params)

        # Setup
        self._filter_widgets = filter_widgets
        metric_names = list(METRIC_LOOKUP.keys())
        self._widgets = [
            pn.widgets.Select(
                    name=f"Metric {idx+1}",
                    options=metric_names,
                    value=metric_names[idx]
                )
        for idx in range(nplots)]

    def __panel__(self):
        return pn.Column("# Empirical CDF", *self._widgets)

    def __iter__(self) -> Generator[ECDFParameters]:
        for idx, w in enumerate(self._widgets):
            metric = METRIC_LOOKUP[w.value]
            yield ECDFParameters(
                index=idx,
                metric=metric,
                metric_label=w.value,
                point_column=metric + "_" + self._filter_widgets.rank + "_point",
                upper_column=metric + "_" + self._filter_widgets.rank + "_upper",
                lower_column=metric + "_" + self._filter_widgets.rank + "_lower"
            )

    def bind(self, function: Callable) -> None:
        """Bind function to widgets."""
        for w in self._widgets:
            pn.bind(function, w.param.value, watch=True, callback_type="ecdf")

class MarkdownView(Viewer):
    """Display Markdown content."""
    def __init__(self, **params) -> None:
        super().__init__(**params)

        self._pane = pn.pane.Markdown(
            "| Site Information |  |  \n| :-- | :-- |  \n|  |  |"
        )

    def update(self, content: str) -> None:
        """Update Markdown content."""
        self._pane.object = content

    def erase(self) -> None:
        """Clear Markdown content."""
        self._pane.object = "| Site Information |  |  \n| :-- | :-- |  \n|  |  |"

    def __panel__(self) -> None:
        return pn.Card(
            self._pane,
            collapsible=False,
            hide_header=True,
            width=350
        )
