"""
Routine operational evaluation layout.
"""
import inspect
from typing import Any

import panel as pn
from panel.viewable import Viewer
import polars as pl

from nwm_explorer.logging.loggers import get_logger
from nwm_explorer.application.api import EvaluationRegistry, Metric, MapLayerName
from nwm_explorer.panes.filters import Filters
from nwm_explorer.panes.site_map import MapLayer, SiteMap, DOMAIN_VIEWS
from nwm_explorer.panes.configuration import ConfigurationPane

class RoutineOperationalEvaluationLayout(Viewer):
    """
    Dashboard main area layout for routine operational evaluations.

    Parameters
    ----------
    registry: EvaluationRegistry
        EvaluationRegistry used throughout dashboard.
    configuration: ConfigurationPane
        ConfigurationPane that specifies dashboard wide options.
    params: any
        Additional keyword arguments passed directly to panel.viewable.Viewer.
    """
    def __init__(
            self,
            registry: EvaluationRegistry,
            configuration: ConfigurationPane,
            **params: dict[str, Any]
            ):
        # Apply parameters
        super().__init__(**params)

        # Get logger
        name = __loader__.name + "." + inspect.currentframe().f_code.co_name
        logger = get_logger(name)
        logger.info("Build routine operational evaluations")

        # Cards
        self.filters = Filters(registry)
        self.site_map = SiteMap(
            layers={
                MapLayerName.metrics: MapLayer(
                    store=registry.scan_evaluation(
                        self.filters.evaluation,
                        self.filters.domain,
                        self.filters.forcing
                    ),
                    color_column="nash_sutcliffe_efficiency_point",
                    custom_data_columns=[
                        "usgs_site_code",
                        "nwm_feature_id"
                    ],
                    custom_data_labels=[
                        "USGS site code",
                        "NWM feature ID"
                    ],
                    colorbar_title=list(Metric)[0],
                    colorbar_limits=(-1.0, 1.0)
                ),
                MapLayerName.site_information: MapLayer(
                    store=pl.scan_parquet(registry.site_information),
                    custom_data_columns=[
                        "site_name",
                        "usgs_site_code",
                        "HUC",
                        "drainage_area",
                        "contributing_drainage_area"
                    ],
                    custom_data_labels=[
                        "Site name",
                        "USGS site code",
                        "HUC",
                        "Drainage Area (sq.mi.)",
                        "Contrib. Drain. Area (sq.mi.)"
                    ],
                    marker_color="rgba(23, 225, 189, 0.75)",
                    marker_size=10
                ),
                MapLayerName.national_inventory_of_dams: MapLayer(
                    store=pl.scan_parquet(registry.national_inventory_of_dams),
                    custom_data_columns=[
                        "riverName",
                        "name",
                        "maxDischarge",
                        "normalStorage",
                        "maxStorage",
                        "drainageArea"
                    ],
                    custom_data_labels=[
                        "River Name",
                        "Dam Name",
                        "Maximum Discharge (CFS)",
                        "Normal Storage (ac-ft)",
                        "Maximum Storage (ac-ft)",
                        "Drainage Area (sq.mi.)"
                    ],
                    marker_color="rgba(255, 141, 0, 0.75)",
                    marker_size=10
                )
            },
            domains=DOMAIN_VIEWS,
            domain_selector=self.filters.domain_selector,
            layer_selector=configuration.layer_selector
        )
        self.histograms = [
            pn.pane.Markdown("Histogram"),
            pn.pane.Markdown("Histogram"),
            pn.pane.Markdown("Histogram"),
            pn.pane.Markdown("Histogram")
            ]
        self.site_information = pn.pane.Markdown("Site information")
        self.hydrograph = pn.pane.Markdown("Hydrograph")
        self.site_metrics = pn.pane.Markdown("Site metrics")

    def __panel__(self) -> pn.Card:
        return pn.Column(
            pn.Row(self.filters, self.site_map, pn.GridBox(*self.histograms, ncols=2)),
            pn.Row(self.site_information, self.hydrograph, self.site_metrics)
        )
