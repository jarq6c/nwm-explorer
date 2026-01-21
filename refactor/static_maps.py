"""CartoPy example."""
from pathlib import Path
from dataclasses import dataclass
from itertools import count

from pydantic import BaseModel

import numpy as np
import pandas as pd
import geopandas as gpd
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.offsetbox import VPacker, TextArea, AnchoredOffsetbox

from shapely.geometry import Polygon
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from cartopy.io.img_tiles import StadiaMapsTiles

COLOR_RAMPS: dict[str, list[str]] = {
    "C0": ["#ca0020", "#f4a582", "#ffffff", "#bababa", "#404040"],
    "C1": ["#a6611a", "#d4af69", "#ded3b7", "#b8d7d2", "#67bfb1", "#018571"]
}
"""Color ramps for markers."""

@dataclass
class MetricStyle:
    """Parameters for styling evaluation metrics."""
    name: str = "Metric"
    slug: str | None = None
    bins: list[float] | None = None
    colors: list[str] | None = None
    labels: dict[str, str] | None = None

    def __post_init__(self) -> None:
        # Apply defaults
        if self.slug is None:
            self.slug = self.name.lower().replace(" ", "_")

        if self.bins is None:
            self.bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

        if self.colors is None:
            self.colors = COLOR_RAMPS["C0"]

        if self.labels is None:
            self.labels = {
                self.colors[i-1]: f"{self.bins[i-1]} to {self.bins[i]}"
                for i in range(1, len(self.bins))
            }

class Config(BaseModel):
    """Configuration options."""
    api_key: str
    default_zoom: int
    zoom_override: dict[str, int]

def main(debug: bool = False) -> None:
    """Main."""
    # Load configuration
    with Path("config.json").open("r", encoding="utf-8") as fi:
        config = Config.model_validate_json(fi.read())

    # Setup tile map
    tiler = StadiaMapsTiles(
        apikey=config.api_key,
        style="stamen_terrain",
        resolution="@2x",
        cache="./data/tiles"
    )

    # Define CRS
    crs = ccrs.Mercator()
    crs_proj4 = crs.proj4_init

    # Load RFC boundaries
    ifile = Path.home() / "Projects/data/rfc_boundaries/rfc_nwm_domain_boundaries_4326.geojson"
    boundaries = gpd.read_file(ifile).to_crs(crs_proj4)

    # Handle OCONUS separately
    conus = boundaries[boundaries["domain"] == "CONUS"]

    # Load evaluation results
    metric_name = "probability_of_detection"
    metric_rank = "max"
    metric_min = 0.0
    metric_max = 1.0
    metric_lead_times = (24, 48)
    model_configuration = "medium_range_mem1"
    threshold = "q85_cfs"
    label = "FY2024-FY2025"
    scores = pl.scan_parquet(
        "/ised/nwm_explorer_data/evaluations",
        hive_schema={
            "label": pl.String,
            "threshold": pl.String,
            "configuration": pl.String
        }
    ).filter(
        pl.col("label") == label,
        pl.col("threshold") == threshold,
        pl.col("configuration") == model_configuration,
        pl.col("lead_time_hours_min") == metric_lead_times[0]
    ).select([
        "nwm_feature_id",
        f"{metric_name}_{metric_rank}_point",
        f"{metric_name}_{metric_rank}_lower",
        f"{metric_name}_{metric_rank}_upper"
    ]).collect().drop_nulls(
        subset=f"{metric_name}_{metric_rank}_point"
    ).with_columns(
        pl.col(f"{metric_name}_{metric_rank}_lower").fill_null(metric_min),
        pl.col(f"{metric_name}_{metric_rank}_upper").fill_null(metric_max),
    ).rename({
        f"{metric_name}_{metric_rank}_point": "score",
        f"{metric_name}_{metric_rank}_lower": "score_lower",
        f"{metric_name}_{metric_rank}_upper": "score_upper"
    }).to_pandas()

    # Compute marker size
    size_coefficient = 400.0
    numerator = -1.0 * scores["score_upper"].sub(scores["score_lower"])
    denominator = scores["score"].mul(2.0)
    scores["marker_size"] = 400.0 * (size_coefficient ** (numerator / denominator))
    scores.loc[scores["marker_size"] < 1.0, "marker_size"] = 1.0

    # Classify scores (assign colors)
    pod_style = MetricStyle("Probability of detection")
    scores["color"] = pd.cut(
        scores["score"],
        bins=pod_style.bins,
        labels=pod_style.colors,
        include_lowest=True
    )

    # Add geometry
    rl = pl.read_parquet(
        "/ised/nwm_explorer_data/routelink.parquet",
        columns=["nwm_feature_id", "latitude", "longitude"]
        ).to_pandas().set_index("nwm_feature_id")
    scores["latitude"] = scores["nwm_feature_id"].map(rl["latitude"])
    scores["longitude"] = scores["nwm_feature_id"].map(rl["longitude"])
    scores["geometry"] = gpd.points_from_xy(
        x=scores["longitude"],
        y=scores["latitude"],
        crs="EPSG:4326"
    )
    scores = gpd.GeoDataFrame(scores).to_crs(crs_proj4)

    # Load river names
    ifiles = [
        Path.home() / (
        "Projects/"
        "data/ne/ne_10m_rivers_lake_centerlines/"
        "ne_10m_rivers_lake_centerlines.shp"
        ),
        Path.home() / (
        "Projects/"
        "data/ne/ne_10m_rivers_north_america/"
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

    # Rivers
    river_lake_centerlines = cfeature.NaturalEarthFeature(
        category="physical",
        name="rivers_lake_centerlines",
        scale="10m",
        facecolor="none",
        edgecolor="#aad3df"
        )
    rivers_na = cfeature.NaturalEarthFeature(
        category="physical",
        name="rivers_north_america",
        scale="10m",
        facecolor="none",
        edgecolor="#aad3df"
        )

    # Lakes
    lakes = cfeature.NaturalEarthFeature(
        category="physical",
        name="lakes",
        scale="10m",
        facecolor="#aad3df",
        edgecolor="#aad3df"
        )
    lakes_na = cfeature.NaturalEarthFeature(
        category="physical",
        name="lakes_north_america",
        scale="10m",
        facecolor="#aad3df",
        edgecolor="#aad3df"
        )

    # Rendering options
    if debug:
        width, height = 6.4, 3.6
        dpi = 150
    else:
        width, height = 12.8, 7.2
        dpi = 300
    aspect_ratio = width / height
    buffer = 0.025

    # Tick label format
    lon_formatter = LongitudeFormatter(number_format=".1f")
    lat_formatter = LatitudeFormatter(number_format=".1f")

    # Generate plots
    for row in conus.itertuples():
        # Reset layer counter
        zlayer = count(1)

        # Create new map
        fig, ax = plt.subplots(
            figsize=(width, height),
            dpi=dpi,
            subplot_kw={"projection": crs}
            )

        # Set extent
        minx, miny, maxx, maxy = row.geometry.bounds
        w, h = maxx - minx, maxy - miny
        w_adjust = (max(w, aspect_ratio * h) - w) / 2.0
        h_adjust = (max(h, w / aspect_ratio) - h) / 2.0
        minx, maxx, miny, maxy = (
            minx-w_adjust-(w*buffer),
            maxx+w_adjust+(w*buffer),
            miny-h_adjust-(h*buffer),
            maxy+h_adjust+(h*buffer)
            )
        ax.set_extent([
                minx-w_adjust-(w*buffer),
                maxx+w_adjust+(w*buffer),
                miny-h_adjust-(h*buffer),
                maxy+h_adjust+(h*buffer)
            ],
            crs=crs)
        ax.set_extent([minx, maxx, miny, maxy], crs=crs)

        # Add tiles
        if debug:
            ax.stock_img()
        else:
            ax.add_image(
                tiler,
                config.zoom_override.get(row.rfc, config.default_zoom),
                interpolation="spline36",
                regrid_shape=4000,
                zorder=next(zlayer)
                )

        # X-ticks
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

        # Add lakes and rivers
        ax.add_feature(lakes, zorder=next(zlayer))
        ax.add_feature(lakes_na, zorder=next(zlayer))
        ax.add_feature(river_lake_centerlines, zorder=next(zlayer))
        ax.add_feature(rivers_na, zorder=next(zlayer))

        # Build shaded polygon
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
        w, h = maxx - minx, maxy - miny
        x_tol, y_tol = 0.05 * w, 0.05 * h
        labels = river_names.sjoin(shaded, how="inner").drop_duplicates(
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
        shaded = shaded.overlay(
            boundaries[boundaries["rfc"] == row.rfc][["geometry"]],
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
            [row.geometry],
            crs=crs,
            alpha=0.25,
            facecolor="white",
            zorder=next(zlayer)
            )

        # Boundary outline
        ax.add_geometries(
            [row.geometry],
            crs=crs,
            facecolor="None",
            linewidth=2,
            linestyle="--",
            zorder=next(zlayer)
            )

        # Add scores
        rfc_scores = scores.sjoin(
            boundaries[boundaries["rfc"] == row.rfc][["geometry"]],
            how="inner")
        for c, df in rfc_scores.groupby("color", observed=True):
            ax.scatter(
                df["geometry"].x,
                df["geometry"].y,
                c=c,
                s=df["marker_size"].values,
                transform=crs,
                edgecolors="black",
                zorder=next(zlayer)
            )
            ax.scatter(
                [],
                [],
                c=c,
                s=100.0,
                transform=crs,
                edgecolors="black",
                label=pod_style.labels[c]
            )

        # Add legend
        ax.legend(
            title=pod_style.name,
            loc="lower left",
            edgecolor="black",
            facecolor="white",
            framealpha=1.0,
            fancybox=False,
            alignment="left"
        )

        # Annotation
        model_region = TextArea(
            "Arkansas-Red Basin",
            textprops={"size": 11, "color": "black", "weight": "bold"}
        )
        title = TextArea(
            "High Flow Evaluation",
            textprops={"size": 10, "color": "black", "weight": "bold"}
        )
        model_version = TextArea(
            "National Water Model v3.0",
            textprops={"size": 9, "color": "black", "weight": "bold"}
        )
        model_config_box = TextArea(
            "Medium Range Forecast (GFS)",
            textprops={"size": 8, "color": "black", "weight": "bold"}
        )
        model_dates = TextArea(
            "2023 Oct 01 to 2025 Sep 30",
            textprops={"size": 7, "color": "black", "weight": "bold"}
        )
        model_lead_times = TextArea(
            f"Lead times: {metric_lead_times[0]} to {metric_lead_times[1]} hours",
            textprops={"size": 7, "color": "black", "weight": "bold"}
        )
        header = VPacker(
            children=[
                model_region,
                title,
                model_version,
                model_config_box,
                model_lead_times,
                model_dates
                ],
            align="right",
            mode="equal"
        )
        header_box = AnchoredOffsetbox(
            loc="upper right",
            frameon=True,
            child=header
        )
        ax.add_artist(header_box)

        # Render map
        if debug:
            plt.show()
            break
        else:
            filename = row.rfc.lower() + "_" + pod_style.slug
            ofile = Path("plots") / f"{filename}_mrf_lt0.png"
            fig.savefig(
                ofile,
                bbox_inches="tight",
                dpi=dpi
            )
        fig.clear()
        break

if __name__ == "__main__":
    main()
