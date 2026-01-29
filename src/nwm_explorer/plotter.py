"""Generate static maps of evaluation results."""
from pathlib import Path
import inspect
from typing import Literal
from dataclasses import dataclass
from collections import namedtuple
from itertools import count
import logging

import numpy as np
import polars as pl
import pandas as pd
import geopandas as gpd

# This import and use statement must occur before any other matplotlib imports
import matplotlib
matplotlib.use("agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.offsetbox import VPacker, TextArea, AnchoredOffsetbox

from shapely.geometry import Polygon
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
    "C0": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "C1": [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0],
    "C2": [0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
}
"""Bins to categorize values."""

METRIC_PLOTTING_COLORS: dict[Metric | CategoricalMetric, str] = {
    Metric.RELATIVE_MEAN_BIAS: "C1",
    Metric.PEARSON_CORRELATION_COEFFICIENT: "C1",
    Metric.NASH_SUTCLIFFE_EFFICIENCY: "C0",
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
    "None": "Threshold: None",
    "q85_cfs": "Threshold: 85th Prc. (NWMv3.0 Retro)",
    "q95_cfs": "Threshold: 85th Prc. (NWMv3.0 Retro)",
    "q99_cfs": "Threshold: 85th Prc. (NWMv3.0 Retro)"
}
"""Mapping from parquet thresholds to descriptive text."""

RANK_LOOKUP: dict[str, str] = {
    "min": "Minimum streamflow",
    "median": "Median streamflow",
    "max": "Maximum streamflow"
}
"""Mapping from resampling functions to descriptive text."""

REGIONS: dict[str, str] = {
    "ABRFC": "Arkansas-Red Basin",
    "APRFC": "Alaska-Pacific",
    "CBRFC": "Colorado Basin",
    "CNRFC": "California-Nevada",
    "LMRFC": "Lower Mississippi",
    "MARFC": "Mid-Atlantic",
    "MBRFC": "Missouri Basin",
    "NCRFC": "North Central",
    "NERFC": "Northeast",
    "NWRFC": "Northwest",
    "OHRFC": "Ohio Basin",
    "SERFC": "Southeast",
    "WGRFC": "West-Gulf"
}
"""Mapping from regional abbreviations to full names."""

HEADER_LOCATION: dict[str, str] = {
    "LMRFC": "lower right",
    "NERFC": "lower right",
    "OHRFC": "lower right",
    "SERFC": "lower right"
}
"""Mapping from regional abbreviations to full names."""

class NoDataError(Exception):
    """Exception raised if no data."""

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
    width:float = 12.8
    height: float = 7.2
    dpi: int = 300
    buffer: float = 0.025

    def __post_init__(self) -> None:
        self.aspect_ratio = self.width / self.height

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

def make_header(
        plot_parameters: PlotData,
        region: str,
        zorder: int,
        loc: str
        )-> AnchoredOffsetbox:
    """Generate header box for map."""
    model_region = TextArea(
        region,
        textprops={"size": 11, "color": "black", "weight": "bold"}
    )
    title = TextArea(
        plot_parameters.title,
        textprops={"size": 7, "color": "black", "weight": "bold"}
    )
    model_version = TextArea(
        plot_parameters.model,
        textprops={"size": 9, "color": "black", "weight": "bold"}
    )
    model_config_box = TextArea(
        plot_parameters.configuration,
        textprops={"size": 8, "color": "black", "weight": "bold"}
    )
    model_dates = TextArea(
        plot_parameters.period,
        textprops={"size": 7, "color": "black", "weight": "bold"}
    )
    model_lead_times = TextArea(
        plot_parameters.lead_times,
        textprops={"size": 7, "color": "black", "weight": "bold"}
    )
    model_threshold = TextArea(
        plot_parameters.threshold,
        textprops={"size": 7, "color": "black", "weight": "bold"}
    )
    header = VPacker(
        children=[
            model_region,
            model_version,
            model_config_box,
            title,
            model_threshold,
            model_lead_times,
            model_dates
            ],
        align="right",
        mode="equal"
    )
    return AnchoredOffsetbox(
        loc=loc,
        frameon=True,
        child=header,
        zorder=zorder
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
        model_title: str = "NWM",
        model_domain: str = "CONUS",
        size_coefficient: float = 200.0,
        minimum_size: float = 20.0
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

    # Check for data
    if data.is_empty():
        raise NoDataError("No data found.")

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
        y=df["latitude"],
        crs="EPSG:4326"
    )

    # Compute marker size
    logger.info("Computing marker sizes")
    numerator = -1.0 * df["upper"].sub(df["lower"])
    denominator = df["point"].mul(2.0)
    df["marker_size"] = size_coefficient * (size_coefficient ** (numerator / denominator))
    df.loc[df["marker_size"] < minimum_size, "marker_size"] = minimum_size
    size_limit = size_coefficient * size_coefficient
    df.loc[df["marker_size"] > size_limit, "marker_size"] = size_limit

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
        grouped_data[PointStyle(*grp)] = gpd.GeoDataFrame(
            pts[["marker_size", "geometry"]], crs="EPSG:4326")

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
        title=RANK_LOOKUP[rank],
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

def plot_single_map(
        plot_parameters: PlotData,
        domain_style: DomainStyle,
        crs: ccrs.Projection,
        crs_proj4: str,
        tiler: StadiaMapsTiles,
        rivers: FeatureSet,
        lon_formatter: LongitudeFormatter,
        lat_formatter: LatitudeFormatter,
        ofile: Path
) -> None:
    """Plot a single map."""
    # Get logger
    name = __loader__.name + "." + inspect.currentframe().f_code.co_name
    logger = get_logger(name)

    logger.info("Checking output file path")
    if ofile.exists():
        logger.info("Skipping %s", ofile)
        return

    logger.info("Domain %s", domain_style.rfc)
    # Reset layer counter
    zlayer = count(1)

    # Create new map
    logger.info("Create new figure")
    fig, ax = plt.subplots(
        figsize=(plot_parameters.width, plot_parameters.height),
        dpi=plot_parameters.dpi,
        subplot_kw={"projection": crs}
        )

    # Convert boundary CRS
    boundary = gpd.GeoDataFrame(
        {"geometry": [domain_style.geometry]},
        crs="EPSG:4326"
        ).to_crs(crs_proj4)

    # Set extent
    logger.info("Set map extent")
    minx, miny, maxx, maxy = boundary["geometry"].iloc[0].bounds
    w, h = maxx - minx, maxy - miny
    w_adjust = (max(w, plot_parameters.aspect_ratio * h) - w) / 2.0
    h_adjust = (max(h, w / plot_parameters.aspect_ratio) - h) / 2.0
    minx, maxx, miny, maxy = (
        minx-w_adjust-(w*plot_parameters.buffer),
        maxx+w_adjust+(w*plot_parameters.buffer),
        miny-h_adjust-(h*plot_parameters.buffer),
        maxy+h_adjust+(h*plot_parameters.buffer)
        )
    ax.set_extent([
            minx-w_adjust-(w*plot_parameters.buffer),
            maxx+w_adjust+(w*plot_parameters.buffer),
            miny-h_adjust-(h*plot_parameters.buffer),
            maxy+h_adjust+(h*plot_parameters.buffer)
        ],
        crs=crs)
    ax.set_extent([minx, maxx, miny, maxy], crs=crs)

    # Add tiles
    logger.info("Add base map tiles")
    ax.add_image(
        tiler,
        7,
        interpolation="spline36",
        regrid_shape=4000,
        zorder=next(zlayer)
        )

    # X-ticks
    logger.info("Set xticks (longitude)")
    xticks = np.linspace(minx, maxx, 7)[1:-1]
    ax.set_xticks(xticks, crs=crs)
    ax.tick_params(
        axis="x",
        which="both",
        direction="in",
        length=3,
        top=True,
        pad=-10,
        labeltop=True,
        labelsize=4
        )
    ax.xaxis.set_major_formatter(lon_formatter)

    # Y-ticks
    logger.info("Set yticks (latitude)")
    yticks = np.linspace(miny, maxy, 7)[1:-1]
    ax.set_yticks(yticks, crs=crs)
    ax.tick_params(
        axis="y",
        which="both",
        direction="in",
        length=3,
        right=True,
        labelrotation=90,
        pad=-10,
        labelright=True,
        labelsize=4
        )
    ax.yaxis.set_major_formatter(lat_formatter)

    # Add extra features
    logger.info("Adding rivers and lakes")
    for fs in rivers.features:
        ax.add_feature(fs, zorder=next(zlayer))

    # Build shaded polygon
    logger.info("Computing background shade")
    shaded = gpd.GeoDataFrame(
        index=[0],
        crs=crs_proj4,
        geometry=[
            Polygon([
            (minx, miny),
            (minx, maxy),
            (maxx, maxy),
            (maxx, miny),
            (minx, miny)
        ])]
    )

    # Add labels
    logger.info("Adding labels")
    w, h = maxx - minx, maxy - miny
    x_tol, y_tol = 0.05 * w, 0.05 * h
    labels = rivers.names.sjoin(shaded, how="inner").drop_duplicates(
        subset="name")
    for l in labels.itertuples():
        # Check boundaries
        if (maxx - l.geometry.x) <= 2.0 * x_tol:
            continue
        if (l.geometry.x - minx) <= x_tol:
            continue
        if (maxy - l.geometry.y) <= y_tol:
            continue
        if (l.geometry.y - miny) <= y_tol:
            continue

        # Add river name
        ax.annotate(
            l.name,
            xy=(l.geometry.x, l.geometry.y),
            transform=crs,
            size=6,
            ha="left",
            va="bottom",
            alpha=0.7,
            zorder=next(zlayer)
        )

    # "Cut" out boundary
    logger.info("Highlight region of interest")
    shaded = shaded.overlay(
        boundary,
        how="symmetric_difference"
    )

    # Add shaded area
    ax.add_geometries(
        shaded.geometry,
        crs=crs,
        alpha=0.25,
        facecolor="black",
        zorder=next(zlayer)
        )

    # Boundary outline
    ax.add_geometries(
        boundary.geometry,
        crs=crs,
        alpha=0.25,
        facecolor="white",
        zorder=next(zlayer)
        )

    # Boundary outline
    ax.add_geometries(
        boundary.geometry,
        crs=crs,
        facecolor="None",
        linewidth=2,
        linestyle="--",
        zorder=next(zlayer)
        )

    # Add metrics
    logger.info("Plotting metrics")
    for (l, c), scores in plot_parameters.data.items():
        # Cut out region of interest
        rfc_scores = scores.to_crs(crs_proj4).sjoin(boundary, how="inner")

        # Check for data
        if rfc_scores.empty:
            raise NoDataError("No data in region")

        ax.scatter(
            rfc_scores["geometry"].x,
            rfc_scores["geometry"].y,
            c=c,
            s=rfc_scores["marker_size"].values,
            transform=crs,
            edgecolors="black",
            zorder=next(zlayer)
        )

        # Add stub for legend
        ax.scatter(
            [],
            [],
            c=c,
            s=100.0,
            transform=crs,
            edgecolors="black",
            label=l
        )

    # Add legend and bring to front
    logger.info("Adding legend")
    l = ax.legend(
        title=plot_parameters.metric,
        loc="lower left",
        edgecolor="black",
        facecolor="white",
        framealpha=1.0,
        fancybox=False,
        alignment="left"
    )
    l.set_zorder(next(zlayer))

    # Add header box
    logger.info("Adding header box")
    ax.add_artist(make_header(
        plot_parameters,
        REGIONS.get(domain_style.rfc, "United States"),
        zorder=next(zlayer),
        loc=HEADER_LOCATION.get(domain_style.rfc, "upper right")
    ))

    # Render
    logger.info("Rendering: %s", ofile)
    fig.savefig(
        ofile,
        bbox_inches="tight",
        dpi=plot_parameters.dpi
    )
    plt.close(fig)

def plot_maps(
        plot_parameters: PlotData,
        enable_logging: bool = True
        ) -> list[Figure]:
    """Map metrics and return a Figure."""
    # Get logger
    name = __loader__.name + "." + inspect.currentframe().f_code.co_name
    logger = get_logger(name)

    # Disable logging (for multiprocessing)
    if not enable_logging:
        logger.setLevel(logging.CRITICAL + 1)

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
    plot_count = count()
    for ds in plot_parameters.domains:
        logger.info("Building output file path")
        ofile = Path("plots") / f"plot_{next(plot_count)}.png"

        try:
            plot_single_map(
                plot_parameters=plot_parameters,
                domain_style=ds,
                crs=crs,
                crs_proj4=crs_proj4,
                tiler=tiler,
                rivers=rivers,
                lon_formatter=lon_formatter,
                lat_formatter=lat_formatter,
                ofile=ofile
            )
        except NoDataError as e:
            logger.info(e)
            continue
