"""Generate and serve exploratory evaluation dashboard."""
from pathlib import Path
import inspect

import polars as pl
import panel as pn
import pandas as pd
import geopandas as gpd
import numpy as np
from panel.template import BootstrapTemplate

from nwm_explorer.logging.logger import get_logger
from nwm_explorer.evaluation.compute import EvaluationRegistry, PREDICTION_RESAMPLING
from nwm_explorer.data.mapping import ModelDomain, ModelConfiguration, Metric, DEFAULT_CENTER, DEFAULT_ZOOM, METRIC_PLOTTING_LIMITS
from nwm_explorer.interfaces.filters import FilteringWidgets, CallbackType, CONFIDENCE_STRINGS
from nwm_explorer.data.routelink import get_routelink_readers
from nwm_explorer.plots.site_map import SiteMap
from nwm_explorer.plots.histogram import Histogram
from nwm_explorer.plots.hydrograph import Hydrograph
from nwm_explorer.data.nwm import get_nwm_reader, generate_reference_dates
from nwm_explorer.data.usgs import get_usgs_reader
from nwm_explorer.plots.barplot import BarPlot
from nwm_explorer.interfaces.site_information import SiteInformationTable
from nwm_explorer.data.usgs_site_info import scan_site_info
from nwm_explorer.interfaces.configuration import ConfigurationWidgets, CMS_FACTOR, MeasurementUnits, INH_FACTOR

pn.extension("plotly")

import plotly.graph_objects as go
from plotly.basedatatypes import BaseTraceType

def load_nid(ifile: Path) -> BaseTraceType:
    if not ifile.exists():
        return go.Scattermap()
    gdf = gpd.read_file(
        ifile,
        columns=[
            "latitude",
            "longitude",
            "name",
            "riverName",
            "maxStorage",
            "normalStorage",
            "maxDischarge",
            "drainageArea"
        ])
    return go.Scattermap(
        marker=dict(size=15),
        showlegend=False,
        name="",
        mode="markers",
        lat=gdf.latitude,
        lon=gdf.longitude,
        visible=True
        # customdata=custom_data,
        # hovertemplate=(
        #     f"<br>{value_label}: "
        #     "%{marker.color:.2f}<br>"
        #     "NWM Feature ID: %{customdata[0]}<br>"
        #     "USGS Site Code: %{customdata[1]}<br>"
        #     "Start Date: %{customdata[2]}<br>"
        #     "End Date: %{customdata[3]}<br>"
        #     "Samples: %{customdata[4]}<br>"
        #     "Longitude: %{lon}<br>"
        #     "Latitude: %{lat}"
    )

    # .rename(columns={
    #     "name": "Dam Name",
    #     "riverName": "River Name",
    #     "drainageArea": "Drainage Area (sq.mi.)",
    #     "maxStorage": "Maximum Storage (ac-ft)",
    #     "normalStorage": "Normal Storage (ac-ft)",
    #     "maxDischarge": "Maximum Discharge (CFS)"
    # })

class Dashboard:
    """Build a dashboard for exploring National Water Model output."""
    def __init__(self, root: Path, title: str):
        # Get logger
        name = __loader__.name + "." + inspect.currentframe().f_code.co_name
        logger = get_logger(name)

        # Setup template
        self.template = BootstrapTemplate(
            title=title,
            collapsed_sidebar=True
        )

        # Setup registry
        registry_file = root / "evaluation_registry.json"
        if registry_file.exists():
            logger.info(f"Reading {registry_file}")
            with registry_file.open("r") as fo:
                self.evaluation_registry = EvaluationRegistry.model_validate_json(fo.read())
        else:
            logger.info(f"No registry found at {registry_file}")
            self.template.main.append(pn.pane.Markdown("# Registry not found. Have you run an evaluation?"))
            return
        
        # Scan evaluation data
        self.routelinks = get_routelink_readers(root)
        self.data: dict[str, dict[ModelDomain, dict[ModelConfiguration, pl.LazyFrame]]] = {}
        self.predictions: dict[str, dict[ModelDomain, dict[ModelConfiguration, pl.LazyFrame]]] = {}
        self.observations: dict[str, dict[ModelDomain, pl.LazyFrame]] = {}
        for label, evaluation_spec in self.evaluation_registry.evaluations.items():
            self.data[label] = {}
            self.predictions[label] = {}
            self.observations[label] = {}
            
            # Collect reference dates
            startDT = pd.Timestamp(evaluation_spec.startDT)
            endDT = pd.Timestamp(evaluation_spec.endDT)
            reference_dates = generate_reference_dates(startDT, endDT)

            for domain, files in evaluation_spec.files.items():
                self.data[label][domain] = {}
                self.predictions[label][domain] = {}

                # Scan observations
                self.observations[label][domain] = get_usgs_reader(
                    root,
                    domain,
                    pd.date_range(startDT-pd.Timedelta("1d"), endDT+pd.Timedelta("10d"), freq="1d").to_list()
                )

                # Scan predictions
                for configuration, ifile in files.items():
                    logger.info(f"Scanning {ifile}")
                    self.data[label][domain][configuration] = pl.scan_parquet(ifile)
                    self.predictions[label][domain][configuration] = get_nwm_reader(
                        root,
                        domain,
                        configuration,
                        reference_dates
                    )
        
        # Load additional map data
        additional_layers: dict[str, BaseTraceType] = {
            "National Inventory of Dams": load_nid(root / "NID.gpkg")
        }

        # Widgets
        self.filters = FilteringWidgets(self.evaluation_registry)
        self.map_center = DEFAULT_CENTER[self.filters.state.domain]
        self.map_zoom = DEFAULT_ZOOM[self.filters.state.domain]
        self.map = SiteMap(
            center=self.map_center,
            zoom=self.map_zoom,
            additional_layers=additional_layers
        )
        self.double_click = False
        self.histogram = Histogram([
            Metric.kling_gupta_efficiency,
            Metric.pearson_correlation_coefficient,
            Metric.relative_mean,
            Metric.relative_standard_deviation
        ])
        self.histogram_columns = [m+c for m in self.histogram.columns for c in CONFIDENCE_STRINGS.values()]
        self.histogram_callbacks = [
            CallbackType.evaluation,
            CallbackType.domain,
            CallbackType.configuration,
            CallbackType.lead_time
        ]
        self.bbox: dict[str, float] | None = None
        self.hydrograph = Hydrograph()
        self.nwm_feature_id = None
        self.usgs_site_code = None
        self.barplot = BarPlot()
        self.site_table = SiteInformationTable(
            scan_site_info(root)
        )
        self.site_options = ConfigurationWidgets(
            layers=list(additional_layers.keys()))

        # Callbacks
        def update_barplot() -> None:
            # Vet call
            if self.nwm_feature_id is None:
                return

            # Current state
            state = self.filters.state
            columns = [state.metric+c for c in CONFIDENCE_STRINGS.values()]

            # Select data
            data = self.data[state.evaluation][state.domain][state.configuration].filter(
                pl.col("nwm_feature_id") == self.nwm_feature_id
            )

            # Deal with forecasts
            if state.configuration in PREDICTION_RESAMPLING:
                columns.append("lead_time_hours_min")
            data = data.select(columns).collect()
            
            # Ignore empty dataframes
            if data.is_empty():
                return

            # Set xdata
            if "lead_time_hours_min" in data:
                xdata = data["lead_time_hours_min"].to_numpy()
            else:
                xdata = [0]
            
            # Update and refresh
            self.barplot.update(
                xdata=xdata,
                ydata=data[state.metric+"_point"].to_numpy(),
                ydata_lower=data[state.metric+"_lower"].to_numpy(),
                ydata_upper=data[state.metric+"_upper"].to_numpy(),
                xlabel="Minimum Lead Time (h)",
                ylabel=state.metric_label
            )
            self.barplot.refresh()

        def update_histogram() -> None:
            # Current state
            state = self.filters.state

            # Select data
            geometry = self.routelinks[state.domain].select(["nwm_feature_id", "latitude", "longitude"])
            data = self.data[state.evaluation][state.domain][state.configuration].join(
                geometry, on="nwm_feature_id", how="left")

            if self.bbox is not None:
                data = data.filter(
                        pl.col("latitude") <= self.bbox["lat_max"],
                        pl.col("latitude") >= self.bbox["lat_min"],
                        pl.col("longitude") <= self.bbox["lon_max"],
                        pl.col("longitude") >= self.bbox["lon_min"]
                    )
            if state.configuration in PREDICTION_RESAMPLING:
                data = data.filter(pl.col("lead_time_hours_min") == state.lead_time)
            data = data.select(self.histogram_columns).collect()
            
            # Ignore empty dataframes
            if data.is_empty():
                return
            
            # Update and refresh
            self.histogram.update(data)
            self.histogram.refresh()

        def update_interface(event, callback_type: CallbackType) -> None:
            # Current state
            state = self.filters.state

            # Reset map view
            if callback_type == CallbackType.double_click:
                self.map_center = DEFAULT_CENTER[state.domain]
                self.map_zoom = DEFAULT_ZOOM[state.domain]
                self.map.layout["map"].update(dict(
                    center=self.map_center,
                    zoom=self.map_zoom
                ))
                self.map.refresh()
                self.bbox = None
                self.double_click = True
                update_histogram()
                return

            # Register zoom
            if callback_type == CallbackType.relayout:
                if self.double_click:
                    self.double_click = False
                    return
                elif "map.center" in event and "map.zoom" in event:
                    self.map_center = event["map.center"]
                    self.map_zoom = event["map.zoom"]

                    # Update histogram
                    self.bbox = {
                        "lat_max": event["map._derived"]["coordinates"][0][1],
                        "lat_min": event["map._derived"]["coordinates"][2][1],
                        "lon_max": event["map._derived"]["coordinates"][1][0],
                        "lon_min": event["map._derived"]["coordinates"][0][0]
                    }
                    update_histogram()
                    return

            # Update domain
            if callback_type == CallbackType.domain:
                self.map_center = DEFAULT_CENTER[state.domain]
                self.map_zoom = DEFAULT_ZOOM[state.domain]
                self.bbox = None
                self.nwm_feature_id = None
                self.usgs_site_code = None

            # Maintain layout
            self.map.layout["map"].update(dict(
                center=self.map_center,
                zoom=self.map_zoom
            ))

            # Select data
            data = self.data[state.evaluation][state.domain][state.configuration]
            geometry = self.routelinks[state.domain].select(["nwm_feature_id", "latitude", "longitude"])

            # Filter data
            value_column = state.metric + state.confidence
            columns = [value_column, "nwm_feature_id", "usgs_site_code", "start_date", "end_date", "sample_size"]
            if state.configuration in PREDICTION_RESAMPLING:
                columns.append("lead_time_hours_min")
                data = data.filter(pl.col("lead_time_hours_min") == state.lead_time)
            data = data.select(columns).join(geometry, on="nwm_feature_id", how="left").with_columns(
                pl.col("start_date").dt.strftime("%Y-%m-%d"),
                pl.col("end_date").dt.strftime("%Y-%m-%d")
            ).collect()
            
            # Update map
            cmin, cmax = METRIC_PLOTTING_LIMITS[state.metric]
            self.map.update(
                values=data[value_column].to_numpy(),
                latitude=data["latitude"].to_numpy(),
                longitude=data["longitude"].to_numpy(),
                value_label=state.metric_label,
                cmin=cmin,
                cmax=cmax,
                custom_data=data.select(columns[1:]).to_pandas()
            )
            
            # Send changes to frontend
            self.map.refresh()

            # Update histogram
            if callback_type in self.histogram_callbacks:
                update_histogram()
            
            # Update barplot
            if callback_type in [CallbackType.configuration, CallbackType.metric]:
                update_barplot()
        pn.bind(
            update_interface,
            self.map.pane.param.doubleclick_data,
            watch=True,
            callback_type=CallbackType.double_click
        )
        pn.bind(
            update_interface,
            self.map.pane.param.relayout_data,
            watch=True,
            callback_type=CallbackType.relayout
        )
        self.filters.register_callback(update_interface)

        def update_site_info() -> None:
            if self.usgs_site_code is None:
                return
            self.site_table.update(self.usgs_site_code)

        def update_hydrograph(event, callback_type: CallbackType) -> None:
            # Vet callback
            if callback_type not in [CallbackType.click, CallbackType.configuration, CallbackType.measurement_units]:
                return

            # Update selected feature
            if callback_type == CallbackType.click:
                data = event["points"][0]["customdata"]
                self.nwm_feature_id = data[0]
                self.usgs_site_code = data[1]
                update_site_info()

            # Check for selected feature
            if self.nwm_feature_id is None:
                return
            if self.usgs_site_code is None: 
                return

            # Current state
            state = self.filters.state
            units = self.site_options.state.units

            # Scan model output
            predictions = self.predictions[state.evaluation][state.domain][state.configuration].filter(
                pl.col("nwm_feature_id") == self.nwm_feature_id).collect()

            # Scan observations
            observations = self.observations[state.evaluation][state.domain].filter(
                pl.col("usgs_site_code") == self.usgs_site_code,
                pl.col("value_time") >= predictions["value_time"].min(),
                pl.col("value_time") <= predictions["value_time"].max(),
            ).collect()
            
            # Apply conversion factors
            ylabel = "CFS"
            if units == MeasurementUnits.cms:
                ylabel = "Streamflow (CMS)"
                observations = observations.with_columns(
                    pl.col("observed").mul(CMS_FACTOR)
                )
                predictions = predictions.with_columns(
                    pl.col("predicted").mul(CMS_FACTOR)
                )
            elif (units == MeasurementUnits.inh) & (~np.isnan(self.site_table.area)):
                ylabel = "Streamflow (inch/h)"
                observations = observations.with_columns(
                    pl.col("observed").mul(INH_FACTOR) / self.site_table.area
                )
                predictions = predictions.with_columns(
                    pl.col("predicted").mul(INH_FACTOR) / self.site_table.area
                )
            elif (units == MeasurementUnits.cfs_sqmi) & (~np.isnan(self.site_table.area)):
                ylabel = "Streamflow (CFS/sq.mi.)"
                observations = observations.with_columns(
                    pl.col("observed") / self.site_table.area
                )
                predictions = predictions.with_columns(
                    pl.col("predicted") / self.site_table.area
                )
            
            # Prepare traces
            x = []
            y = []
            n = []

            # Observation traces
            x.append(observations["value_time"].to_numpy())
            y.append(observations["observed"].to_numpy())
            n.append(f"USGS-{self.usgs_site_code}")

            # Prediction traces
            if state.configuration in PREDICTION_RESAMPLING:
                for (rt,), df in predictions.partition_by("reference_time", maintain_order=True, as_dict=True).items():
                    x.append(df["value_time"].to_numpy())
                    y.append(df["predicted"].to_numpy())
                    n.append(rt.strftime("%Y-%m-%d %HZ"))
            else:
                x.append(predictions["value_time"].to_numpy())
                y.append(predictions["predicted"].to_numpy())
                n.append("Analysis")

            # Update hydrograph
            self.hydrograph.update_data(
                x=x,
                y=y,
                names=n,
                ylabel=ylabel
            )
            self.hydrograph.refresh()

            # Update barplot
            update_barplot()
        pn.bind(
            update_hydrograph,
            self.map.pane.param.click_data,
            watch=True,
            callback_type=CallbackType.click
        )
        self.filters.register_callback(update_hydrograph)
        self.site_options.register_callback(update_hydrograph)

        # Layout
        controls = pn.Column(
            self.filters.servable(),
            self.site_table.servable()
        )
        top_display = pn.Row(
            self.map.servable(),
            self.histogram.servable()
        )
        bottom_display = pn.Row(
            self.hydrograph.servable(),
            self.barplot.servable()
        )
        display = pn.Column(
            top_display,
            bottom_display
        )
        self.template.main.append(
            pn.Row(
                controls,
                display
        ))
        self.template.sidebar.append(
            self.site_options.servable()
        )
    
    def servable(self) -> BootstrapTemplate:
        return self.template

def generate_dashboard(
        root: Path,
        title: str
        ) -> BootstrapTemplate:
    return Dashboard(root, title).servable()

def generate_dashboard_closure(
        root: Path,
        title: str
        ) -> BootstrapTemplate:
    def closure():
        return generate_dashboard(root, title)
    return closure

def serve_dashboard(
        root: Path,
        title: str
        ) -> None:
    # Slugify title
    slug = title.lower().replace(" ", "-")

    # Serve
    endpoints = {
        slug: generate_dashboard_closure(root, title)
    }
    pn.serve(endpoints)
