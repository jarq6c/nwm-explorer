"""Generate and serve exploratory evaluation dashboard."""
from pathlib import Path
import inspect

import polars as pl
import panel as pn
import pandas as pd
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

class Dashboard:
    """Build a dashboard for exploring National Water Model output."""
    def __init__(self, root: Path, title: str):
        # Get logger
        name = __loader__.name + "." + inspect.currentframe().f_code.co_name
        logger = get_logger(name)

        # Setup template
        self.template = BootstrapTemplate(title=title)

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
        for label, evaluation_spec in self.evaluation_registry.evaluations.items():
            self.data[label] = {}
            self.predictions[label] = {}
            
            # Scan predictions
            startDT = pd.Timestamp(evaluation_spec.startDT)
            endDT = pd.Timestamp(evaluation_spec.endDT)
            reference_dates = generate_reference_dates(startDT, endDT)

            for domain, files in evaluation_spec.files.items():
                self.data[label][domain] = {}
                self.predictions[label][domain] = {}
                for configuration, ifile in files.items():
                    logger.info(f"Scanning {ifile}")
                    self.data[label][domain][configuration] = pl.scan_parquet(ifile)
                    self.predictions[label][domain][configuration] = get_nwm_reader(
                        root,
                        domain,
                        configuration,
                        reference_dates
                    )

        # Widgets
        self.filters = FilteringWidgets(self.evaluation_registry)
        self.map_center = DEFAULT_CENTER[self.filters.state.domain]
        self.map_zoom = DEFAULT_ZOOM[self.filters.state.domain]
        self.map = SiteMap(
            center=self.map_center,
            zoom=self.map_zoom
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

        # Callbacks
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

        def update_hydrograph(event) -> None:
            # Current state
            state = self.filters.state

            # Feature ID
            data = event["points"][0]["customdata"]
            feature_id = data[0]

            # Scan model output
            predictions = self.predictions[state.evaluation][state.domain][state.configuration].filter(
                pl.col("nwm_feature_id") == feature_id)

            print(predictions.head().collect())
        pn.bind(
            update_hydrograph,
            self.map.pane.param.click_data,
            watch=True
        )

        # Layout
        controls = pn.Column(self.filters.servable())
        top_display = pn.Row(
            self.map.servable(),
            self.histogram.servable()
        )
        bottom_display = pn.Row(
            self.hydrograph.servable()
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
