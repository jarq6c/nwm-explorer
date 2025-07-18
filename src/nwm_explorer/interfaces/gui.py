"""Generate and serve exploratory evaluation dashboard."""
from pathlib import Path
import inspect

import polars as pl
import panel as pn
from panel.template import BootstrapTemplate

from nwm_explorer.logging.logger import get_logger
from nwm_explorer.evaluation.compute import EvaluationRegistry, PREDICTION_RESAMPLING
from nwm_explorer.data.mapping import ModelDomain, ModelConfiguration
from nwm_explorer.interfaces.filters import FilteringWidgets, CallbackType
from nwm_explorer.data.routelink import get_routelink_readers
from nwm_explorer.plots.site_map import SiteMap, METRIC_PLOTTING_LIMITS

class Histogram:
    def __init__(self):
        pass

    def servable(self) -> pn.Card:
        return pn.Card(
            pn.pane.Markdown("HISTOGRAM"),
            collapsible=False,
            hide_header=True
        )

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
                evaluation_registry = EvaluationRegistry.model_validate_json(fo.read())
        else:
            logger.info(f"No registry found at {registry_file}")
            self.template.main.append(pn.pane.Markdown("# Registry not found. Have you run an evaluation?"))
            return
        
        # Scan evaluation data
        self.routelinks = get_routelink_readers(root)
        self.data: dict[str, dict[ModelDomain, dict[ModelConfiguration, pl.LazyFrame]]] = {}
        for label, evaluation_spec in evaluation_registry.evaluations.items():
            self.data[label] = {}
            for domain, files in evaluation_spec.files.items():
                self.data[label][domain] = {}
                for configuration, ifile in files.items():
                    logger.info(f"Scanning {ifile}")
                    self.data[label][domain][configuration] = pl.scan_parquet(ifile)
        
        # Widgets
        self.filters = FilteringWidgets(evaluation_registry)
        self.map = SiteMap()
        self.histogram = Histogram()
        self.state = self.filters.state

        # Callbacks
        def update_map(event, callback_type: CallbackType) -> None:
            if event is None:
                return
            
            # Limit number of state updates
            if self.state == self.filters.state:
                return
            
            # Update state
            self.state = self.filters.state

            # Select data
            data = self.data[self.state.evaluation][self.state.domain][self.state.configuration]
            geometry = self.routelinks[self.state.domain].select(["nwm_feature_id", "latitude", "longitude"])

            # Filter data
            value_column = self.state.metric + self.state.confidence
            columns = [value_column, "nwm_feature_id", "usgs_site_code", "start_date", "end_date", "sample_size"]
            if self.state.configuration in PREDICTION_RESAMPLING:
                columns.append("lead_time_hours_min")
                data = data.filter(pl.col("lead_time_hours_min") == self.state.lead_time)
            data = data.select(columns).join(geometry, on="nwm_feature_id", how="left").with_columns(
                pl.col("start_date").dt.strftime("%Y-%m-%d"),
                pl.col("end_date").dt.strftime("%Y-%m-%d")
            ).collect()
            
            # Update map
            cmin, cmax = METRIC_PLOTTING_LIMITS[self.state.metric]
            self.map.update(
                values=data[value_column].to_numpy(),
                latitude=data["latitude"].to_numpy(),
                longitude=data["longitude"].to_numpy(),
                value_label=self.state.metric_label,
                cmin=cmin,
                cmax=cmax,
                domain=self.state.domain,
                custom_data=data.select(columns[1:]).to_pandas()
            )
            self.map.refresh()
        self.filters.register_callback(update_map)
        
        # def update_histogram(event, callback_type: CallbackType) -> None:
        #     if event is None:
        #         return
            
        #     if callback_type == CallbackType.domain:
        #         print(event)
            
        #     if callback_type == CallbackType.relayout:
        #         if "map.center" not in event:
        #             return
        #         print(event)
        # self.filters.register_callback(update_histogram)
        # pn.bind(update_histogram, self.map.relayout_data, watch=True,
        #     callback_type=CallbackType.relayout)

        # Layout
        self.template.main.append(
            pn.Row(
                self.filters.servable(),
                self.map.servable(),
                self.histogram.servable()
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
