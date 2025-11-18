"""Dashboard objects."""
from pathlib import Path
from typing import Any

import polars as pl
import panel as pn
import pandas as pd
from panel.template import MaterialTemplate
from panel.viewable import Viewer

from .nwm import nwm_site_generator
from .evaluate import load_metrics, scan_evaluations, load_site_metrics
from .routelink import download_routelink
from .usgs import usgs_site_generator, load_site_information
from .views import (FilterWidgets, MapView, TimeSeriesView, BarPlot, ECDFPlot,
    MarkdownView)
from .constants import METRIC_PLOTTING_LIMITS, CONFIGURATION_LINE_TYPE

class Dashboard(Viewer):
    """Build a dashboard for exploring National Water Model output."""
    def __init__(self, root: Path, title: str, **params):
        super().__init__(**params)

        # Open routelink
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

        site_map = MapView()
        hydrograph = TimeSeriesView()
        barplot = BarPlot()
        ecdf = ECDFPlot()
        site_information = MarkdownView()
        site_column_mapping = {
            "monitoring_location_name": "Name",
            "monitoring_location_number": "Site code",
            "hydrologic_unit_code": "HUC",
            "site_type": "Site type",
            "drainage_area": "Drainage area (sq.mi.)",
            "contributing_drainage_area": "Contrib. drain. area (sq.mi.)"
        }

        def handle_map_click(event, callback_type: str) -> None:
            if event is None:
                return

            # Check for state
            if not site_map.click_data:
                # Clear barplot
                barplot.erase()

                # Clear hydrograph
                hydrograph.erase_data(
                    xrange=(
                        data_ranges["observed_value_time_min"],
                        data_ranges["observed_value_time_max"]
                        )
                )

                # Clear site info
                site_information.erase()
                return

            # Parse custom data
            metadata = site_map.click_data["customdata"]
            nwm_feature_id = metadata[0]
            usgs_site_code = metadata[1]

            # Update site info
            if callback_type == "click":
                # Build site table
                info = "| Site Information |  |  \n| :-- | :-- |  \n"
                for series in load_site_information(root, usgs_site_code,
                    rename=site_column_mapping).iter_columns():
                    info += f"| **{series.name}** | {series.item(0)} |  \n"
                url = f"https://waterdata.usgs.gov/monitoring-location/USGS-{usgs_site_code}/"
                info += "| **Monitoring page** | "
                info += f'<a href="{url}" target="_blank">Open new tab</a> |'
                site_information.update(info)

            # Retrieve metrics
            metric_data = load_site_metrics(
                root=root,
                label=filter_widgets.label,
                configuration=filter_widgets.configuration,
                metric=filter_widgets.metric,
                nwm_feature_id=nwm_feature_id,
                rank=filter_widgets.rank,
                cache=True
            )

            # These filters do not update hydrograph or barplot
            if callback_type in ["lead_time", "significant"]:
                return

            # Clear barplot
            barplot.erase()

            # Update barplot
            barplot.update(
                xdata=metric_data["lead_time_hours_min"],
                ydata=metric_data[filter_widgets.point_column].to_numpy(),
                ydata_lower=metric_data[filter_widgets.lower_column].to_numpy(),
                ydata_upper=metric_data[filter_widgets.upper_column].to_numpy(),
                xlabel="Minimum Lead Time (h)",
                ylabel=filter_widgets.metric_label
            )

            # These filters do not update hydrograph
            if callback_type in ["rank", "metric"]:
                return

            # Clear hydrograph
            hydrograph.erase_data(
                xrange=(
                    data_ranges["observed_value_time_min"],
                    data_ranges["observed_value_time_max"]
                    )
            )

            # Stream observations
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
                mode = CONFIGURATION_LINE_TYPE.get(
                    filter_widgets.configuration, "lines"
                )
                hydrograph.append_traces(trace_data, mode=mode)
        site_map.bind_click(handle_map_click)

        def handle_filter_updates(event: str, callback_type: str) -> None:
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

            # Update CDF
            if site_map.viewport:
                sorted_data = data.filter(
                    pl.col("latitude") <= site_map.viewport["lat_max"],
                    pl.col("latitude") >= site_map.viewport["lat_min"],
                    pl.col("longitude") <= site_map.viewport["lon_max"],
                    pl.col("longitude") >= site_map.viewport["lon_min"]
                ).sort(filter_widgets.point_column, descending=False)
            else:
                sorted_data = data.sort(filter_widgets.point_column, descending=False)
            ecdf.update(
                xdata=sorted_data[filter_widgets.point_column].to_numpy(),
                xdata_lower=sorted_data[filter_widgets.lower_column].to_numpy(),
                xdata_upper=sorted_data[filter_widgets.upper_column].to_numpy(),
                xlabel=filter_widgets.metric_label,
                xrange=METRIC_PLOTTING_LIMITS[filter_widgets.metric]
            )

            # Update barplot and hydrograph
            handle_map_click(0, callback_type)
        handle_filter_updates(filter_widgets.label, "intialize")
        filter_widgets.bind(handle_filter_updates)

        def handle_relayout(event: dict[str, Any], callback_type: str) -> None:
            # Ignore non-calls
            if event is None or callback_type is None:
                return

            # Ignore non-zooms
            if not site_map.viewport:
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
            ).filter(
                pl.col("latitude") <= site_map.viewport["lat_max"],
                pl.col("latitude") >= site_map.viewport["lat_min"],
                pl.col("longitude") <= site_map.viewport["lon_max"],
                pl.col("longitude") >= site_map.viewport["lon_min"]
            ).sort(
                filter_widgets.point_column, descending=False
            )

            # Update ECDF
            ecdf.update(
                xdata=data[filter_widgets.point_column].to_numpy(),
                xdata_lower=data[filter_widgets.lower_column].to_numpy(),
                xdata_upper=data[filter_widgets.upper_column].to_numpy(),
                xlabel=filter_widgets.metric_label,
                xrange=METRIC_PLOTTING_LIMITS[filter_widgets.metric]
            )
        site_map.bind_relayout(handle_relayout)

        # Setup template
        self.template = MaterialTemplate(
            title=title,
            collapsed_sidebar=True
        )

        # Layout
        self.template.main.append(pn.Row(
            pn.Column(filter_widgets, site_information),
            pn.Column(site_map, hydrograph),
            pn.Column(ecdf, barplot)
        ))

    def __panel__(self) -> MaterialTemplate:
        return self.template
