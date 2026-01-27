"""Generate static maps of evaluation results."""
from pathlib import Path
import inspect
from typing import Literal
from dataclasses import dataclass
from collections import namedtuple
from itertools import count

import polars as pl
import pandas as pd
import geopandas as gpd

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# from shapely.geometry import Polygon
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from cartopy.io.img_tiles import StadiaMapsTiles

from nwm_explorer.evaluate import scan_evaluations
from nwm_explorer.routelink import download_routelink
from nwm_explorer.logger import get_logger
from nwm_explorer.constants import (ModelConfiguration, EvaluationMetric,
    METRIC_PLOTTING_LIMITS, CONFIGURATION_LOOKUP, METRIC_LOOKUP,
    GROUP_SPECIFICATIONS, Metric, CategoricalMetric)

COLOR_RAMPS: dict[str, list[str]] = {
    "C0": ["#ca0020", "#f4a582", "#ffffff", "#bababa", "#404040"],
    "C1": ["#a6611a", "#d4af69", "#ded3b7", "#b8d7d2", "#67bfb1", "#018571"],
    "C2": ["#a6611a", "#d4af69", "#ded3b7", "#b8d7d2", "#67bfb1", "#018571"]
}
"""Color ramps for markers."""

BINS: dict[str, list[float]] = {
    "C0": [0.0, 0.2, 0.4, 0.6, 1.0],
    "C1": [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0],
    "C2": [0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
}
"""Bins to categorize values."""

METRIC_PLOTTING_COLORS: dict[Metric | CategoricalMetric, str] = {
    Metric.RELATIVE_MEAN_BIAS: "C1",
    Metric.PEARSON_CORRELATION_COEFFICIENT: "C1",
    Metric.NASH_SUTCLIFFE_EFFICIENCY: "C1",
    Metric.RELATIVE_MEAN: "C2",
    Metric.RELATIVE_STANDARD_DEVIATION: "C2",
    Metric.RELATIVE_MEDIAN: "C2",
    Metric.RELATIVE_MINIMUM: "C2",
    Metric.RELATIVE_MAXIMUM: "C2",
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

PointStyle = namedtuple("PointStyle", ["label", "color"])
"""Named tuple for storing ('label', 'color')."""
PointStyle.__doc__ = "Named tuple for storing point style information."

DomainStyle = namedtuple("DomainStyle", ["rfc", "domain", "geometry"])
"""Named tuple for storing ('rfc', 'domain', 'geometry')."""
DomainStyle.__doc__ = "Named tuple for storing domain style information."

@dataclass
class PlotData:
    """Parameters used to generate map of metrics."""
    data: dict[PointStyle, gpd.GeoDataFrame]
    title: str
    model: str
    configuration: ModelConfiguration
    metric: EvaluationMetric
    period: str
    stadia_api_key: str
    domains: list[DomainStyle]
    map_directory: Path
    threshold: str = "None"
    lead_times: str | None = None
    width:float = 6.4
    height: float = 3.6
    dpi: int = 300
    buffer: float = 0.025

@dataclass
class FeatureSet:
    """Supplmental features to plot on map."""
    names: gpd.GeoDataFrame
    features: list[cfeature.NaturalEarthFeature]

def handle_rivers_and_lakes(
        plot_parameters: PlotData,
        crs_proj4: str
        ) -> FeatureSet:
    """Handle sources for river and lake geometry and labels."""
    # Load river names
    ifiles = [
        plot_parameters.map_directory / (
        "ne/ne_10m_rivers_lake_centerlines/"
        "ne_10m_rivers_lake_centerlines.shp"
        ),
        plot_parameters.map_directory / (
        "ne/ne_10m_rivers_north_america/"
        "ne_10m_rivers_north_america.shp"
        ),]
    river_names = pd.concat([
        gpd.read_file(f, columns=["name"]).dropna().to_crs(crs_proj4)
        for f in ifiles], ignore_index=True)

    # Place label at downstream end
    river_names["geometry"] = river_names["geometry"].interpolate(1.0)

    # Remove overlapping labels (keep first)
    drop = []
    for row in river_names.itertuples():
        distance = river_names["geometry"].distance(row.geometry)
        check = river_names[distance <= 60_000.0]
        if check["name"].count() > 1:
            # Keep first
            for i in check.index[1:]:
                if i not in drop:
                    drop.append(i)
    river_names = river_names[~river_names.index.isin(drop)]

    # Build feature set
    features: list[cfeature.NaturalEarthFeature] = []

    # Rivers
    features.append(cfeature.NaturalEarthFeature(
        category="physical",
        name="rivers_lake_centerlines",
        scale="10m",
        facecolor="none",
        edgecolor="#aad3df"
        ))
    features.append(cfeature.NaturalEarthFeature(
        category="physical",
        name="rivers_north_america",
        scale="10m",
        facecolor="none",
        edgecolor="#aad3df"
        ))

    # Lakes
    features.append(cfeature.NaturalEarthFeature(
        category="physical",
        name="lakes",
        scale="10m",
        facecolor="#aad3df",
        edgecolor="#aad3df"
        ))
    features.append(cfeature.NaturalEarthFeature(
        category="physical",
        name="lakes_north_america",
        scale="10m",
        facecolor="#aad3df",
        edgecolor="#aad3df"
        ))
    
    return FeatureSet(
        names=river_names,
        features=features
    )

def plot_preprocess(
        root: Path,
        label: str,
        configuration: ModelConfiguration,
        metric: EvaluationMetric,
        threshold: str,
        lead_time_hours_min: int,
        api_key: str,
        rank: Literal["min", "median", "max"],
        title: str = "Evaluation",
        model_title: str = "NWM",
        model_domain: str = "CONUS",
        size_coefficient: float = 400.0
    ) -> PlotData:
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
    logger.info("Computing marker sizes")
    numerator = -1.0 * df["upper"].sub(df["lower"])
    denominator = df["point"].mul(2.0)
    df["marker_size"] = size_coefficient * (size_coefficient ** (numerator / denominator))
    df.loc[df["marker_size"] < 1.0, "marker_size"] = 1.0

    # Classify scores (assign colors)
    logger.info("Assigning marker colors")
    point_color = df["point"].copy()
    point_color[point_color <= cmin] = cmin * 0.99
    point_color[point_color >= cmax] = cmax * 0.99
    colors = COLOR_RAMPS[METRIC_PLOTTING_COLORS[metric]]
    bins = BINS[METRIC_PLOTTING_COLORS[metric]]
    df["marker_color"] = pd.cut(
        point_color,
        bins=bins,
        labels=colors,
        right=True
    )
    bin_labels = [f"{bins[i-1]:.2f} to {bins[i]:.2f}" for i in range(1, len(bins))]
    df["bin_label"] = pd.cut(
        point_color,
        bins=bins,
        labels=bin_labels,
        right=True,
        precision=1
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

    # Group data by value
    logger.info("Grouping points")
    cols = ["bin_label", "marker_color"]
    df = df[cols+["marker_size", "geometry"]]
    grouped_data: dict[PointStyle, gpd.GeoDataFrame] = {}
    for grp, pts in df.groupby(cols, observed=True):
        grouped_data[PointStyle(*grp)] = pts[["marker_size", "geometry"]]

    # Load RFC boundaries
    logger.info("Loading RFC boundaries")
    ifile = root / "map_layers/rfc_nwm_domain_boundaries_4326.geojson"
    boundaries = gpd.read_file(ifile)

    # Set domain
    boundaries = boundaries[boundaries["domain"] == model_domain]

    # Build parameters
    logger.info("Listing subregions")
    domains: list[DomainStyle] = []
    for row in boundaries.itertuples(index=False):
        domains.append(DomainStyle(*row))

    logger.info("Configuring plot parameters")
    return PlotData(
        data=grouped_data,
        title=title,
        model=model_title,
        configuration=CONFIGURATION_LOOKUP[configuration],
        metric=METRIC_LOOKUP_REVERSE[metric],
        period=period,
        lead_times=lead_times,
        threshold=THRESHOLD_LOOKUP[threshold],
        domains=domains,
        stadia_api_key=api_key,
        map_directory=root/"map_layers"
    )

def plot_map(plot_parameters: PlotData) -> Figure:
    """Map metrics and return a Figure."""
    # Get logger
    name = __loader__.name + "." + inspect.currentframe().f_code.co_name
    logger = get_logger(name)

    # Setup tile map
    logger.info("Setting tiles")
    tiler = StadiaMapsTiles(
        apikey=plot_parameters.stadia_api_key,
        style="stamen_terrain",
        resolution="@2x",
        cache=plot_parameters.map_directory/"tiles"
    )

    # Define CRS
    logger.info("Setting CRS")
    crs = ccrs.Mercator()
    crs_proj4 = crs.proj4_init

    # Deal with rivers and lakes
    logger.info("Gathering river and lake geometry")
    rivers = handle_rivers_and_lakes(
        plot_parameters=plot_parameters,
        crs_proj4=crs_proj4
    )

    # Tick label format
    lon_formatter = LongitudeFormatter(number_format=".1f")
    lat_formatter = LatitudeFormatter(number_format=".1f")

    # Generate plots
    logger.info("Gathering plots")
    for ds in plot_parameters.domains:
        logger.info("Domain %s", ds.rfc)

        # Reset layer counter
        zlayer = count(1)

        # Create new map
        fig, ax = plt.subplots(
            figsize=(plot_parameters.width, plot_parameters.height),
            dpi=plot_parameters.dpi,
            subplot_kw={"projection": crs}
            )

        # # Plot
        # logger.info("Plotting metrics")
        # for (l, c), gdf in plot_parameters.data.items():
        #     print(l, c)
        #     break

        # Render map
        # filename = row.rfc.lower() + "_" + pod_style.slug
        # ofile = Path("plots") / f"{filename}_mrf_lt0.png"
        # fig.savefig(
        #     ofile,
        #     bbox_inches="tight",
        #     dpi=plot_parameters.dpi
        # )
        fig.clear()
        break
