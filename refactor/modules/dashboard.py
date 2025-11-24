"""Dashboard objects."""
from typing import Any

import numpy as np
import polars as pl
import panel as pn
import pandas as pd
from panel.template import MaterialTemplate
from panel.viewable import Viewer

from .nwm import nwm_site_generator
from .evaluate import load_metrics, scan_evaluations, load_site_metrics
from .routelink import download_routelink
from .usgs import usgs_site_generator, load_site_information
from .views import (FilterWidgets, MapView, TimeSeriesView, BarPlot, ECDFMatrix,
    MarkdownView, ECDFSelector)
from .constants import (METRIC_PLOTTING_LIMITS, CONFIGURATION_LINE_TYPE, SITE_COLUMN_MAPPING,
    MeasurementUnits)
from .options import StreamflowOptions, compute_conversion_factor
from .configuration import Configuration

class Dashboard(Viewer):
    """Build a dashboard for exploring National Water Model output."""
    def __init__(self, configuration: Configuration, **params):
        super().__init__(**params)
        # Handle configuration
        root = configuration.root
        title = configuration.title

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

        site_map = MapView(map_layers=configuration.map_layers)
        hydrograph = TimeSeriesView()
        barplot = BarPlot()
        ecdf = ECDFMatrix(nplots=4, ncols=2)
        ecdf_filters = ECDFSelector(nplots=4, filter_widgets=filter_widgets)
        site_information = MarkdownView()
        streamflow_options = StreamflowOptions()

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
                    rename=SITE_COLUMN_MAPPING).iter_columns():
                    info += f"| **{series.name}** | {series.item(0)} |  \n"
                url = f"https://waterdata.usgs.gov/monitoring-location/USGS-{usgs_site_code}/"
                info += "| **Monitoring page** | "
                info += f'<a href="{url}" target="_blank">Open new tab</a> |'
                site_information.update(info)

            # These filters do not update hydrograph or barplot
            if callback_type in ["lead_time", "significant"]:
                return

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

            # Unit conversion
            conversion_factor = compute_conversion_factor(
                root, usgs_site_code, streamflow_options.measurement_units
            )

            # Update y-axis title
            hydrograph.set_yaxis_title(f"Streamflow ({streamflow_options.measurement_units})")

            # Stream observations
            dataframes = []
            for df in usgs_site_generator(
                root=root,
                usgs_site_code=usgs_site_code,
                start_time=data_ranges["observed_value_time_min"],
                end_time=data_ranges["observed_value_time_max"],
                cache=True
            ):
                # Check for data
                if df.is_empty():
                    continue

                # Append data
                dataframes.append(df)
                observations = pl.concat(dataframes)

                # Resample
                if streamflow_options.measurement_units in [
                    MeasurementUnits.INCHES_PER_HOUR,
                    MeasurementUnits.CUMULATIVE_INCHES_PER_HOUR
                ]:
                    observations = observations.group_by_dynamic(
                        "value_time", every="1h"
                    ).agg(pl.col("observed_cfs").mean())

                # Apply conversion
                if conversion_factor != 1.0:
                    observations = observations.with_columns(
                        pl.col("observed_cfs").mul(conversion_factor)
                    )

                # Accumulate
                if streamflow_options.measurement_units in [
                    MeasurementUnits.CUMULATIVE_INCHES_PER_HOUR]:
                    observations = observations.with_columns(
                        pl.col("observed_cfs").cum_sum()
                    )

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
                # Add date string column
                df = df.with_columns(
                    pl.col(
                        "reference_time"
                    ).dt.strftime(
                        "Issued: %Y-%m-%d %HZ"
                    ).alias("datetime_string")
                )

                # Add each reference time
                trace_data = []
                for rt in df["datetime_string"].unique():
                    # Extract predictions
                    predictions = df.filter(pl.col("datetime_string") == rt)

                    # Apply conversion
                    if conversion_factor != 1.0:
                        predictions = predictions.with_columns(
                            pl.col("predicted_cfs").mul(conversion_factor)
                        )

                    # Accumulate
                    # TODO These accumulations need to be intialized/pinned to observations
                    if streamflow_options.measurement_units in [
                        MeasurementUnits.CUMULATIVE_INCHES_PER_HOUR]:
                        predictions = predictions.with_columns(
                            pl.col("predicted_cfs").cum_sum()
                        )

                    # Add trace data
                    trace_data.append((
                        predictions["value_time"].to_numpy(),
                        predictions["predicted_cfs"].to_numpy()
                        rt
                    ))

                # Add to plot
                mode = CONFIGURATION_LINE_TYPE.get(
                    filter_widgets.configuration, "lines"
                )
                hydrograph.append_traces(trace_data, mode=mode)
        site_map.bind_click(handle_map_click)

        def handle_relayout(event: dict[str, Any], callback_type: str) -> None:
            # Ignore non-calls
            if event is None or callback_type is None:
                return

            # Update each plot
            for p in ecdf_filters:
                # Retrieve data
                data = load_metrics(
                    root=root,
                    label=filter_widgets.label,
                    configuration=filter_widgets.configuration,
                    metric=p.metric,
                    lead_time_hours_min=filter_widgets.lead_time,
                    rank=filter_widgets.rank,
                    additional_columns=(
                        "nwm_feature_id",
                        "usgs_site_code",
                        "sample_size"
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

                # Sort data
                if site_map.viewport:
                    sorted_data = data.filter(
                        pl.col("latitude") <= site_map.viewport["lat_max"],
                        pl.col("latitude") >= site_map.viewport["lat_min"],
                        pl.col("longitude") <= site_map.viewport["lon_max"],
                        pl.col("longitude") >= site_map.viewport["lon_min"]
                    ).sort(p.point_column, descending=False)
                else:
                    sorted_data = data.sort(p.point_column, descending=False)

                # Update ECDF
                ecdf.update(
                    index=p.index,
                    xdata=sorted_data[p.point_column].to_numpy(),
                    xdata_lower=sorted_data[p.lower_column].to_numpy(),
                    xdata_upper=sorted_data[p.upper_column].to_numpy(),
                    xlabel=p.metric_label,
                    xrange=METRIC_PLOTTING_LIMITS[p.metric]
                )
        site_map.bind_relayout(handle_relayout)
        ecdf_filters.bind(handle_relayout)

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

            # Update ECDF
            handle_relayout(0, callback_type)

            # Update barplot and hydrograph
            handle_map_click(0, callback_type)
        filter_widgets.bind(handle_filter_updates)

        # Handle streamflow options
        def handle_streamflow_options(event: str, callback_type: str) -> None:
            if event is None:
                return

            # Change scale
            if callback_type == "axis_scale":
                hydrograph.set_axis_type(streamflow_options.axis_scale)
        streamflow_options.bind(handle_streamflow_options)

        # Initialize states
        handle_filter_updates(filter_widgets.label, "intialize")
        handle_relayout(0, "intialize")

        # Setup template
        self.template = MaterialTemplate(
            title=title,
            collapsed_sidebar=True
        )

        # Main area
        controls = pn.Column(filter_widgets, site_information)
        content = pn.Column(
            pn.Row(site_map, ecdf),
            pn.Row(hydrograph, barplot)
        )
        self.template.main.append(pn.Row(
            controls,
            content
        ))

        # Sidebar
        self.template.sidebar.append(ecdf_filters)
        self.template.sidebar.append(streamflow_options)
        self.template.sidebar.append(site_map.map_layer_selector)

    def __panel__(self) -> MaterialTemplate:
        return self.template
