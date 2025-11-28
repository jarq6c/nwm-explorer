"""Dashboard objects."""
from typing import Any
from datetime import datetime

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
    MeasurementUnits, ModelConfiguration)
from .options import StreamflowOptions, compute_conversion_factor
from .configuration import Configuration

def generate_usgs_url(
        usgs_site_code: str,
        start_datetime: datetime,
        end_datetime: datetime
    ) -> str:
    """Returns URL to USGS monitoring location page."""
    # Base monitoring location URL
    url = f"https://waterdata.usgs.gov/monitoring-location/USGS-{usgs_site_code}/#"

    # parameters
    params = {
        "dataTypeId": "continuous-00060-0",
        "showFieldMeasurements": "true",
        "startDT": start_datetime.strftime("%Y-%m-%dT%H:%M"),
        "endDT": end_datetime.strftime("%Y-%m-%dT%H:%M")
    }

    # Finish building URL
    for k, v in params.items():
        url += f"&{k}={v}"
    return url

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
        self.usgs_site_code: str | None = None
        self.nwm_feature_id: int | None = None

        site_map = MapView(map_layers=configuration.map_layers)
        hydrograph = TimeSeriesView()
        barplot = BarPlot()
        ecdf = ECDFMatrix(nplots=4, ncols=2)
        ecdf_filters = ECDFSelector(nplots=4, filter_widgets=filter_widgets)
        site_information = MarkdownView()
        streamflow_options = StreamflowOptions()

        def draw_hydrograph(event, callback_type: str) -> None:
            # Ignore non-event
            if event is None:
                return

            # Ignore axis scale change
            if callback_type in ["axis_scale"]:
                return

            # Check for site selection
            if self.usgs_site_code is None:
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
                root, self.usgs_site_code, streamflow_options.measurement_units
            )

            # Update y-axis title
            hydrograph.set_yaxis_title(f"Streamflow ({streamflow_options.measurement_units})")

            # Stream observations
            dataframes = []
            for df in usgs_site_generator(
                root=root,
                usgs_site_code=self.usgs_site_code,
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
                    ).agg(
                        pl.col("observed_cfs").mean()
                    ).upsample(
                        time_column="value_time", every="1h"
                    ).fill_null(
                        strategy="forward"
                    )

                # Apply conversion
                if conversion_factor != 1.0:
                    observations = observations.with_columns(
                        pl.col("observed_cfs").mul(conversion_factor)
                    )

                # Accumulate
                if streamflow_options.measurement_units in [
                    MeasurementUnits.CUMULATIVE_INCHES_PER_HOUR
                ]:
                    observations = observations.with_columns(
                        pl.col("observed_cfs").cum_sum()
                    )

                # Replace data
                hydrograph.update_trace(
                    xdata=observations["value_time"].to_numpy(),
                    ydata=observations["observed_cfs"].to_numpy(),
                    name=f"USGS-{self.usgs_site_code}"
                )

            # Stream predictions
            accumulated_value = 0.0
            for df in nwm_site_generator(
                root=root,
                configuration=filter_widgets.configuration,
                nwm_feature_id=self.nwm_feature_id,
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
                ).sort(["reference_time", "value_time"])

                # Add each reference time
                trace_data = []
                for rt in df["datetime_string"].unique(maintain_order=True):
                    # Extract predictions
                    predictions = df.filter(pl.col("datetime_string") == rt)

                    # Apply conversion
                    if conversion_factor != 1.0:
                        predictions = predictions.with_columns(
                            pl.col("predicted_cfs").mul(conversion_factor)
                        )

                    # Accumulate
                    if streamflow_options.measurement_units in [
                        MeasurementUnits.CUMULATIVE_INCHES_PER_HOUR
                    ]:
                        # Distinguish between analysis and forecast
                        if filter_widgets.configuration in [
                            ModelConfiguration.ANALYSIS_ASSIM_EXTEND_ALASKA_NO_DA,
                            ModelConfiguration.ANALYSIS_ASSIM_EXTEND_NO_DA,
                            ModelConfiguration.ANALYSIS_ASSIM_HAWAII_NO_DA,
                            ModelConfiguration.ANALYSIS_ASSIM_PUERTO_RICO_NO_DA
                        ]:
                            predictions = predictions.with_columns(
                                pl.col("predicted_cfs").cum_sum().add(accumulated_value)
                            )
                            accumulated_value = predictions["predicted_cfs"].max()
                        else:
                            intial_time = predictions["value_time"].min() - pl.duration(hours=1)
                            vals = observations.filter(
                                pl.col("value_time") == intial_time
                            )["observed_cfs"]
                            if vals.is_empty():
                                accumulated_value = 0.0
                            else:
                                accumulated_value = vals.item(0)
                            predictions = predictions.with_columns(
                                pl.col("predicted_cfs").cum_sum().add(accumulated_value)
                            )

                    # Add trace data
                    trace_data.append((
                        predictions["value_time"].to_numpy(),
                        predictions["predicted_cfs"].to_numpy(),
                        rt
                    ))

                # Add to plot
                mode = CONFIGURATION_LINE_TYPE.get(
                    filter_widgets.configuration, "lines"
                )
                hydrograph.append_traces(trace_data, mode=mode)

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
                self.usgs_site_code = None
                self.nwm_feature_id = None
                return

            # Parse custom data
            metadata = site_map.click_data["customdata"]
            self.nwm_feature_id = metadata[0]
            self.usgs_site_code = metadata[1]

            # Update site info
            if callback_type == "click":
                # Build site table
                info = "| Site Information |  |  \n| :-- | :-- |  \n"
                for series in load_site_information(root, self.usgs_site_code,
                    rename=SITE_COLUMN_MAPPING).iter_columns():
                    info += f"| **{series.name}** | {series.item(0)} |  \n"
                url = generate_usgs_url(
                    self.usgs_site_code,
                    data_ranges["observed_value_time_min"],
                    data_ranges["observed_value_time_max"]
                    )
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
                nwm_feature_id=self.nwm_feature_id,
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

            # Update hydrograph
            draw_hydrograph(event, callback_type)
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
                return

            # Apply options (i.e. change measurement units)
            draw_hydrograph(event, callback_type)
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
