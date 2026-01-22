"""Generate static maps of evaluation results."""
from pathlib import Path
import inspect
from typing import Literal
from dataclasses import dataclass

import numpy as np
import polars as pl
import pandas as pd
import geopandas as gpd

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from nwm_explorer.evaluate import scan_evaluations
from nwm_explorer.routelink import download_routelink
from nwm_explorer.logger import get_logger
from nwm_explorer.constants import (ModelConfiguration, EvaluationMetric,
    METRIC_PLOTTING_LIMITS, CONFIGURATION_LOOKUP, METRIC_LOOKUP,
    GROUP_SPECIFICATIONS, Metric, CategoricalMetric)

COLOR_RAMPS: dict[str, list[str]] = {
    "C0": ["#ca0020", "#f4a582", "#ffffff", "#bababa", "#404040"],
    "C1": ["#a6611a", "#d4af69", "#ded3b7", "#b8d7d2", "#67bfb1", "#018571"]
}
"""Color ramps for markers."""

METRIC_PLOTTING_COLORS: dict[Metric | CategoricalMetric, str] = {
    Metric.RELATIVE_MEAN_BIAS: "C1",
    Metric.PEARSON_CORRELATION_COEFFICIENT: "C1",
    Metric.NASH_SUTCLIFFE_EFFICIENCY: "C1",
    Metric.RELATIVE_MEAN: "C1",
    Metric.RELATIVE_STANDARD_DEVIATION: "C1",
    Metric.RELATIVE_MEDIAN: "C1",
    Metric.RELATIVE_MINIMUM: "C1",
    Metric.RELATIVE_MAXIMUM: "C1",
    Metric.KLING_GUPTA_EFFICIENCY: "C1",
    CategoricalMetric.PROBABILITY_OF_DETECTION: "C0",
    CategoricalMetric.PROBABILITY_OF_FALSE_DETECTION: "C0",
    CategoricalMetric.PROBABILITY_OF_FALSE_ALARM: "C0",
    CategoricalMetric.THREAT_SCORE: "C0",
    CategoricalMetric.FREQUENCY_BIAS: "C1",
    CategoricalMetric.PERCENT_CORRECT: "C0",
    CategoricalMetric.EQUITABLE_THREAT_SCORE: "C0",
}
"""Mapping from Metrics to plotting color ramp codes."""

METRIC_LOOKUP_REVERSE: dict[EvaluationMetric, str] = {v: k for k, v in METRIC_LOOKUP.items()}
"""Reverse lookup from metric to metric label."""

THRESHOLD_LOOKUP: dict[str, str] = {
    "None": "None",
    "q85_cfs": "NWM Retro Daily Max Streamflow 85th Percentile",
    "q95_cfs": "NWM Retro Daily Max Streamflow 95th Percentile",
    "q99_cfs": "NWM Retro Daily Max Streamflow 99th Percentile"
}
"""Mapping from parquet thresholds to descriptive text."""

@dataclass
class PlotParameters:
    """Parameters used to generate map of metrics."""
    data: gpd.GeoDataFrame
    title: str
    model: str
    configuration: ModelConfiguration
    metric: EvaluationMetric
    period: str
    threshold: str = "None"
    domain: str | None = None
    lead_times: str | None = None

def plot_preprocess(
        root: Path,
        label: str,
        configuration: ModelConfiguration,
        metric: EvaluationMetric,
        threshold: str,
        lead_time_hours_min: int,
        rank: Literal["min", "median", "max"],
        title: str = "Evaluation",
        model_title: str = "NWM",
        model_domain: str = "CONUS",
        size_coefficient: float = 400.0
    ) -> PlotParameters:
    """
    Load and preprocess evaluation results for plotting.
    """
    # Get logger
    name = __loader__.name + "." + inspect.currentframe().f_code.co_name
    logger = get_logger(name)

    # Look-up plotting limits to fill missing CIs for plotting
    logger.info("Looking up metric limits for %s", metric)
    cmin, cmax = METRIC_PLOTTING_LIMITS[str(metric)]

    # Load metrics, drop or fill missing values
    logger.info("Loading...")
    logger.info("Label: %s", label)
    logger.info("Configuration: %s", configuration)
    logger.info("Lead time: %d", lead_time_hours_min)
    logger.info("Threshold: %s", threshold)
    data = scan_evaluations(
        root
    ).filter(
        pl.col("label") == label,
        pl.col("configuration") == configuration,
        pl.col("lead_time_hours_min") == lead_time_hours_min,
        pl.col("threshold") == threshold
    ).select(
        "nwm_feature_id",
        f"{metric}_{rank}_lower",
        f"{metric}_{rank}_point",
        f"{metric}_{rank}_upper",
        "reference_time_min",
        "reference_time_max"
    ).collect().drop_nulls(
        subset=f"{metric}_{rank}_point"
    ).with_columns(
        pl.col(f"{metric}_{rank}_lower").fill_null(cmin),
        pl.col(f"{metric}_{rank}_upper").fill_null(cmax)
    ).with_columns(
        pl.when(
            pl.col(f"{metric}_{rank}_lower") > pl.col(f"{metric}_{rank}_point")
            ).then(
                pl.col(f"{metric}_{rank}_point")
                ).otherwise(pl.col(f"{metric}_{rank}_lower")
        ).alias(f"{metric}_{rank}_lower")
    ).rename(
        {
            f"{metric}_{rank}_lower": "lower",
            f"{metric}_{rank}_point": "point",
            f"{metric}_{rank}_upper": "upper"
        }
    )

    # Get routelink
    rl = download_routelink(root=root).select(
        "nwm_feature_id",
        "latitude",
        "longitude"
    ).collect()

    # Add coordinates
    logger.info("Applying coordinates")
    df = data.with_columns(
        latitude=pl.col("nwm_feature_id").replace_strict(
            old=rl["nwm_feature_id"].implode(),
            new=rl["latitude"].implode()
        ),
        longitude=pl.col("nwm_feature_id").replace_strict(
            old=rl["nwm_feature_id"].implode(),
            new=rl["longitude"].implode()
        )
    ).to_pandas()

    # Build geometry
    logger.info("Georeferencing evaluation metrics")
    df["geometry"] = gpd.points_from_xy(
        x=df["longitude"],
        y=df["latitude"]
    )

    # Compute marker size
    numerator = -1.0 * df["upper"].sub(df["lower"])
    denominator = df["point"].mul(2.0)
    df["marker_size"] = size_coefficient * (size_coefficient ** (numerator / denominator))
    df.loc[df["marker_size"] < 1.0, "marker_size"] = 1.0

    # Classify scores (assign colors)
    colors = COLOR_RAMPS[METRIC_PLOTTING_COLORS[metric]]
    print(colors)
    print(np.linspace(cmin, cmax, len(colors)+1))
    # FIXME 6 bins doesn't divide up nicely
    quit()
    df["marker_color"] = pd.cut(
        df["point"],
        bins=np.linspace(cmin, cmax, len(colors)+1),
        labels=colors,
        include_lowest=True
    )

    # Determine simulation period
    logger.info("Extracting period of record")
    start: pd.Timestamp = df["reference_time_min"].min()
    end: pd.Timestamp = df["reference_time_max"].max()
    period = (
        start.strftime("%Y %b %d") + " to " + end.strftime("%Y %b %d")
    )

    # Build lead times
    logger.info("Inspecting lead times")
    specs = GROUP_SPECIFICATIONS[configuration]
    if specs.lead_time_hours_max == 0:
        lead_times = None
    elif lead_time_hours_min == specs.lead_time_hours_max:
        lead_times = f"Lead time: {lead_time_hours_min} hours"
    else:
        l = lead_time_hours_min + specs.window_interval
        lead_times = f"Lead times: {lead_time_hours_min} to {l} hours"

    # Build parameters
    logger.info("Configuring plot parameters")
    return PlotParameters(
        data=df[[
            "point",
            "lower",
            "upper",
            "marker_size",
            "marker_color",
            "geometry"
        ]],
        title=title,
        model=model_title,
        configuration=CONFIGURATION_LOOKUP[configuration],
        metric=METRIC_LOOKUP_REVERSE[metric],
        period=period,
        lead_times=lead_times,
        threshold=THRESHOLD_LOOKUP[threshold],
        domain=model_domain
    )

def plot_map(plot_parameters: PlotParameters) -> Figure:
    """Map metrics and return a Figure."""
    # Get logger
    name = __loader__.name + "." + inspect.currentframe().f_code.co_name
    logger = get_logger(name)

    logger.info("Plotting metrics")
    print(plot_parameters)
