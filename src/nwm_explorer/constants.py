"""Constants used throughout the application."""
from pathlib import Path
from enum import StrEnum
from dataclasses import dataclass
from typing import TypedDict, Callable

import polars as pl
import us
from plotly.basedatatypes import BaseTraceType
import plotly.graph_objects as go
import numpy as np
import numpy.typing as npt
import pandas as pd

URLBuilder = Callable[[pd.Timestamp], list[str]]
"""A function that takes a pandas.Timestamp and returns a list of URLs."""

class ModelDomainDisplay(StrEnum):
    """Symbols for model domain display."""
    CONUS = "[CONUS] "
    ALASKA = "[Alaska] "
    HAWAII = "[Hawaii] "
    PUERTO_RICO = "[Puerto Rico] "

class ModelDomain(StrEnum):
    """Model domain."""
    ALASKA = "alaska"
    HAWAII = "hawaii"
    CONUS = "conus"
    PUERTO_RICO = "puertorico"

class ModelConfiguration(StrEnum):
    """Model configurations."""
    ANALYSIS_ASSIM_EXTEND_ALASKA_NO_DA = "analysis_assim_extend_alaska_no_da"
    ANALYSIS_ASSIM_EXTEND_NO_DA = "analysis_assim_extend_no_da"
    ANALYSIS_ASSIM_HAWAII_NO_DA = "analysis_assim_hawaii_no_da"
    ANALYSIS_ASSIM_PUERTO_RICO_NO_DA = "analysis_assim_puertorico_no_da"
    MEDIUM_RANGE_MEM_1 = "medium_range_mem1"
    MEDIUM_RANGE_BLEND = "medium_range_blend"
    MEDIUM_RANGE_NO_DA = "medium_range_no_da"
    MEDIUM_RANGE_ALASKA_MEM_1 = "medium_range_alaska_mem1"
    MEDIUM_RANGE_BLEND_ALASKA = "medium_range_blend_alaska"
    MEDIUM_RANGE_ALASKA_NO_DA = "medium_range_alaska_no_da"
    SHORT_RANGE = "short_range"
    SHORT_RANGE_ALASKA = "short_range_alaska"
    SHORT_RANGE_HAWAII = "short_range_hawaii"
    SHORT_RANGE_HAWAII_NO_DA = "short_range_hawaii_no_da"
    SHORT_RANGE_PUERTO_RICO = "short_range_puertorico"
    SHORT_RANGE_PUERTO_RICO_NO_DA = "short_range_puertorico_no_da"

class Metric(StrEnum):
    """Symbols for common metrics."""
    NASH_SUTCLIFFE_EFFICIENCY = "nash_sutcliffe_efficiency"
    RELATIVE_MEAN_BIAS = "relative_mean_bias"
    PEARSON_CORRELATION_COEFFICIENT = "pearson_correlation_coefficient"
    RELATIVE_MEAN = "relative_mean"
    RELATIVE_MEDIAN = "relative_median"
    RELATIVE_MINIMUM = "relative_minimum"
    RELATIVE_MAXIMUM = "relative_maximum"
    RELATIVE_STANDARD_DEVIATION = "relative_standard_deviation"
    KLING_GUPTA_EFFICIENCY = "kling_gupta_efficiency"

class SiteTypeSlug(StrEnum):
    """Machine-friendly site types."""
    STREAM = "stream"
    CANAL = "canal"
    DITCH = "ditch"
    LAKE = "lake"
    TIDAL = "tidal"
    SPRING = "spring"
    DIVERSION = "diversion"
    ESTUARY = "estuary"
    TUNNEL = "tunnel"
    FIELD = "field"
    STORM_SEWER = "storm_sewer"
    COMBINED_SEWER = "combined_sewer"
    OUTFALL = "outfall"

@dataclass
class SiteType:
    """
    Dataclass for storing USGS site type details.

    Attributes
    ----------
    code: str
        Site type code.
    name: str
        Short name.
    long_name: str
        Long name.
    slug: SiteTypeSlug
        Machine-friendly representation.
    """
    code: str
    name: str
    long_name: str
    slug: SiteTypeSlug

@dataclass
class NWMGroupSpecification:
    """
    A dataclass that holds specifications for time-based polars groupby
    operations.
    """
    index_column: str = "value_time"
    group_by_columns: list[str] | None = None
    select_columns: list[str] | None = None
    sort_columns: list[str] | None = None
    window_interval: int = 24
    state_code: str | None = None
    lead_time_hours_max: int = 0

    def __post_init__(self) -> None:
        if self.group_by_columns is None:
            self.group_by_columns = ["nwm_feature_id", "reference_time"]

        if self.select_columns is None:
            self.select_columns = self.group_by_columns + [self.index_column, "predicted_cfs"]

        if self.sort_columns is None:
            self.sort_columns = self.group_by_columns + [self.index_column]

ROUTELINK_URL: str = (
    "https://www.hydroshare.org/resource"
    "/1fe9975004ce4b5097d41939afa14f84/data/contents/RouteLinks.tar.gz"
)
"""URL to RouteLink CSV tarball."""

ROUTELINK_PARQUET: Path = Path("routelink.parquet")
"""Default path to polars-compatible RouteLink parquet file used by application."""

ROUTELINK_FILENAMES: dict[ModelDomain, str] = {
    ModelDomain.ALASKA: "RouteLink_AK.csv",
    ModelDomain.CONUS: "RouteLink_CONUS.csv",
    ModelDomain.HAWAII: "RouteLink_HI.csv",
    ModelDomain.PUERTO_RICO: "RouteLink_PRVI.csv"
}
"""Mapping from domains to routelink files names."""

GOOGLE_CLOUD_BUCKET_URL: str = "https://storage.googleapis.com/national-water-model/"
"""National Water Model Google Cloud Storage bucket."""

DOMAIN_LOOKUP: dict[ModelConfiguration, ModelDomainDisplay] = {
    ModelConfiguration.ANALYSIS_ASSIM_EXTEND_NO_DA: ModelDomainDisplay.CONUS,
    ModelConfiguration.MEDIUM_RANGE_MEM_1: ModelDomainDisplay.CONUS,
    ModelConfiguration.MEDIUM_RANGE_BLEND: ModelDomainDisplay.CONUS,
    ModelConfiguration.MEDIUM_RANGE_NO_DA: ModelDomainDisplay.CONUS,
    ModelConfiguration.SHORT_RANGE: ModelDomainDisplay.CONUS,
    ModelConfiguration.ANALYSIS_ASSIM_EXTEND_ALASKA_NO_DA: ModelDomainDisplay.ALASKA,
    ModelConfiguration.MEDIUM_RANGE_ALASKA_MEM_1: ModelDomainDisplay.ALASKA,
    ModelConfiguration.MEDIUM_RANGE_BLEND_ALASKA: ModelDomainDisplay.ALASKA,
    ModelConfiguration.MEDIUM_RANGE_ALASKA_NO_DA: ModelDomainDisplay.ALASKA,
    ModelConfiguration.SHORT_RANGE_ALASKA: ModelDomainDisplay.ALASKA,
    ModelConfiguration.ANALYSIS_ASSIM_HAWAII_NO_DA: ModelDomainDisplay.HAWAII,
    ModelConfiguration.SHORT_RANGE_HAWAII: ModelDomainDisplay.HAWAII,
    ModelConfiguration.SHORT_RANGE_HAWAII_NO_DA: ModelDomainDisplay.HAWAII,
    ModelConfiguration.ANALYSIS_ASSIM_PUERTO_RICO_NO_DA: ModelDomainDisplay.PUERTO_RICO,
    ModelConfiguration.SHORT_RANGE_PUERTO_RICO: ModelDomainDisplay.PUERTO_RICO,
    ModelConfiguration.SHORT_RANGE_PUERTO_RICO_NO_DA: ModelDomainDisplay.PUERTO_RICO
}
"""Mapping from ModelConfiguration to ModelDomainDisplay."""

CONFIGURATION_LOOKUP: dict[ModelConfiguration, str] = {
    ModelConfiguration.ANALYSIS_ASSIM_EXTEND_NO_DA: "Extended Analysis & Assimilation"
                                                    " (MRMS/Stage IV, No-DA)",
    ModelConfiguration.MEDIUM_RANGE_MEM_1: "Medium Range Deterministic (GFS)",
    ModelConfiguration.MEDIUM_RANGE_BLEND: "Medium Range Deterministic (NBM)",
    ModelConfiguration.MEDIUM_RANGE_NO_DA: "Medium Range Deterministic (GFS, No-DA)",
    ModelConfiguration.SHORT_RANGE: "Short Range (HRRR)",
    ModelConfiguration.ANALYSIS_ASSIM_EXTEND_ALASKA_NO_DA: "Extended Analysis & Assimilation"
                                                           " (MRMS/Stage IV, No-DA)",
    ModelConfiguration.MEDIUM_RANGE_ALASKA_MEM_1: "Medium Range Deterministic (GFS)",
    ModelConfiguration.MEDIUM_RANGE_BLEND_ALASKA: "Medium Range Deterministic (NBM)",
    ModelConfiguration.MEDIUM_RANGE_ALASKA_NO_DA: "Medium Range Deterministic (GFS, No-DA)",
    ModelConfiguration.SHORT_RANGE_ALASKA: "Short Range (HRRR)",
    ModelConfiguration.ANALYSIS_ASSIM_HAWAII_NO_DA: "Analysis & Assimilation (MRMS, No-DA)",
    ModelConfiguration.SHORT_RANGE_HAWAII: "Short Range (WRF-ARW)",
    ModelConfiguration.SHORT_RANGE_HAWAII_NO_DA: "Short Range (WRF-ARW, No-DA)",
    ModelConfiguration.ANALYSIS_ASSIM_PUERTO_RICO_NO_DA: "Analysis & Assimilation (MRMS, No-DA)",
    ModelConfiguration.SHORT_RANGE_PUERTO_RICO: "Short Range (WRF-ARW)",
    ModelConfiguration.SHORT_RANGE_PUERTO_RICO_NO_DA: "Short Range (WRF-ARW, No-DA)"
}
"""Mapping from ModelConfiguration to pretty strings."""

GROUP_SPECIFICATIONS: dict[ModelConfiguration, NWMGroupSpecification] = {
    ModelConfiguration.ANALYSIS_ASSIM_EXTEND_ALASKA_NO_DA: NWMGroupSpecification(
        group_by_columns=["nwm_feature_id"],
        state_code="ak",
        select_columns=["nwm_feature_id", "reference_time", "value_time", "predicted_cfs"]
    ),
    ModelConfiguration.MEDIUM_RANGE_ALASKA_MEM_1: NWMGroupSpecification(
        state_code="ak",
        lead_time_hours_max=240
    ),
    ModelConfiguration.MEDIUM_RANGE_BLEND_ALASKA: NWMGroupSpecification(
        state_code="ak",
        lead_time_hours_max=240
    ),
    ModelConfiguration.MEDIUM_RANGE_ALASKA_NO_DA: NWMGroupSpecification(
        state_code="ak",
        lead_time_hours_max=240
    ),
    ModelConfiguration.SHORT_RANGE_ALASKA: NWMGroupSpecification(
        window_interval=5,
        state_code="ak",
        lead_time_hours_max=45
    ),
    ModelConfiguration.ANALYSIS_ASSIM_HAWAII_NO_DA: NWMGroupSpecification(
        group_by_columns=["nwm_feature_id"],
        state_code="hi",
        select_columns=["nwm_feature_id", "reference_time", "value_time", "predicted_cfs"]
    ),
    ModelConfiguration.SHORT_RANGE_HAWAII: NWMGroupSpecification(
        window_interval=6,
        state_code="hi",
        lead_time_hours_max=48
    ),
    ModelConfiguration.SHORT_RANGE_HAWAII_NO_DA: NWMGroupSpecification(
        window_interval=6,
        state_code="hi",
        lead_time_hours_max=48
    ),
    ModelConfiguration.ANALYSIS_ASSIM_PUERTO_RICO_NO_DA: NWMGroupSpecification(
        group_by_columns=["nwm_feature_id"],
        state_code="pr",
        select_columns=["nwm_feature_id", "reference_time", "value_time", "predicted_cfs"]
    ),
    ModelConfiguration.SHORT_RANGE_PUERTO_RICO: NWMGroupSpecification(
        window_interval=6,
        state_code="pr",
        lead_time_hours_max=48
    ),
    ModelConfiguration.SHORT_RANGE_PUERTO_RICO_NO_DA: NWMGroupSpecification(
        window_interval=6,
        state_code="pr",
        lead_time_hours_max=48
    ),
    ModelConfiguration.ANALYSIS_ASSIM_EXTEND_NO_DA: NWMGroupSpecification(
        group_by_columns=["nwm_feature_id"],
        select_columns=["nwm_feature_id", "reference_time", "value_time", "predicted_cfs"]
    ),
    ModelConfiguration.MEDIUM_RANGE_MEM_1: NWMGroupSpecification(lead_time_hours_max=240),
    ModelConfiguration.MEDIUM_RANGE_BLEND: NWMGroupSpecification(lead_time_hours_max=240),
    ModelConfiguration.MEDIUM_RANGE_NO_DA: NWMGroupSpecification(lead_time_hours_max=240),
    ModelConfiguration.SHORT_RANGE: NWMGroupSpecification(
        window_interval=6,
        lead_time_hours_max=18
    )
}
"""Mapping from ModelConfiguration to group-by specifications."""

METRIC_LOOKUP: dict[str, Metric] = {
    "Kling-Gupta efficiency": Metric.KLING_GUPTA_EFFICIENCY,
    "Nash-Sutcliffe efficiency": Metric.NASH_SUTCLIFFE_EFFICIENCY,
    "Relative mean": Metric.RELATIVE_MEAN,
    "Relative standard deviation": Metric.RELATIVE_STANDARD_DEVIATION,
    "Pearson correlation coefficient": Metric.PEARSON_CORRELATION_COEFFICIENT,
    "Relative mean bias": Metric.RELATIVE_MEAN_BIAS,
    "Relative median": Metric.RELATIVE_MEDIAN,
    "Relative minimum": Metric.RELATIVE_MINIMUM,
    "Relative maximum": Metric.RELATIVE_MAXIMUM
}
"""Mapping from pretty strings to evaluation Metric."""

RANK_LOOKUP: dict[str, str] = {
    "Median": "median",
    "Minimum": "min",
    "Maximum": "max"
}
"""Mapping from pretty strings to column label components. 'Rank' refers to the
aggregation method used to resample streamflow.
"""

DEFAULT_ZOOM: dict[ModelDomainDisplay, int] = {
    ModelDomainDisplay.ALASKA: 5,
    ModelDomainDisplay.CONUS: 3,
    ModelDomainDisplay.HAWAII: 6,
    ModelDomainDisplay.PUERTO_RICO: 8
}
"""Default map zoom for each domain."""

DEFAULT_CENTER: dict[ModelDomainDisplay, dict[str, float]] = {
    ModelDomainDisplay.ALASKA: {"lat": 60.84683, "lon": -149.05659},
    ModelDomainDisplay.CONUS: {"lat": 38.83348, "lon": -93.97612},
    ModelDomainDisplay.HAWAII: {"lat": 21.24988, "lon": -157.59606},
    ModelDomainDisplay.PUERTO_RICO: {"lat": 18.21807, "lon": -66.32802}
}
"""Default map center for each domain."""

METRIC_PLOTTING_LIMITS: dict[Metric, tuple[float, float]] = {
    Metric.RELATIVE_MEAN_BIAS: (-1.0, 1.0),
    Metric.PEARSON_CORRELATION_COEFFICIENT: (-1.0, 1.0),
    Metric.NASH_SUTCLIFFE_EFFICIENCY: (-1.0, 1.0),
    Metric.RELATIVE_MEAN: (0.0, 2.0),
    Metric.RELATIVE_STANDARD_DEVIATION: (0.0, 2.0),
    Metric.RELATIVE_MEDIAN: (0.0, 2.0),
    Metric.RELATIVE_MINIMUM: (0.0, 2.0),
    Metric.RELATIVE_MAXIMUM: (0.0, 2.0),
    Metric.KLING_GUPTA_EFFICIENCY: (-1.0, 1.0)
}
"""Mapping from Metrics to plotting limits (cmin, cmax)."""

METRIC_SIGNIFICANCE_THRESHOLD: dict[Metric, float] = {
    Metric.RELATIVE_MEAN_BIAS: 0.0,
    Metric.PEARSON_CORRELATION_COEFFICIENT: 0.0,
    Metric.NASH_SUTCLIFFE_EFFICIENCY: 0.0,
    Metric.RELATIVE_MEAN: 1.0,
    Metric.RELATIVE_STANDARD_DEVIATION: 1.0,
    Metric.RELATIVE_MEDIAN: 1.0,
    Metric.RELATIVE_MINIMUM: 1.0,
    Metric.RELATIVE_MAXIMUM: 1.0,
    Metric.KLING_GUPTA_EFFICIENCY: 0.0
}
"""
Mapping from Metrics to a value that if it falls outside the 95% confidence interval,
indicates 'statistical significance.'
"""

LRU_CACHE_SIZES: dict[str, int] = {
    "evaluations": 10,
    "nwm": 9,
    "usgs": 25,
    "pairs": 60
}
"""Maximum size of functools.lru_cache."""

SUBDIRECTORIES: dict[str, str] = {
    "evaluations": "evaluations",
    "nwm": "nwm",
    "usgs": "usgs",
    "site_table": "site_table",
    "pairs": "pairs"
}
"""Subdirectories that indicates root of parquet stores."""

NWIS_BASE_URL: str = (
    "https://waterservices.usgs.gov/nwis/iv/"
    "?format=json&siteStatus=all&parameterCd=00060"
)
"""NWIS IV API returning json and all site statuses."""

MONITORING_LOCATION_BASE_URL: str = (
    "https://api.waterdata.usgs.gov/ogcapi/v0/collections/monitoring-locations"
    "/items?f=json&lang=en-US&limit=10000&skipGeometry=false&offset=0"
    "&agency_code=USGS&site_type_code="
)
"""USGS monitoring location API returning geojson."""

SITE_TYPES: list[SiteType] = [
    SiteType("ST", "Stream", "Stream", SiteTypeSlug.STREAM),
    SiteType("ST-CA", "Canal", "Canal", SiteTypeSlug.CANAL),
    SiteType("ST-DCH", "Ditch", "Ditch", SiteTypeSlug.DITCH),
    SiteType("ST-TS", "Tidal SW", "Tidal stream", SiteTypeSlug.TIDAL),
    SiteType("LK", "Lake", "Lake, Reservoir, Impoundment", SiteTypeSlug.LAKE),
    SiteType("SP", "Spring", "Spring", SiteTypeSlug.SPRING),
    SiteType("FA-DV", "Diversion", "Diversion", SiteTypeSlug.DIVERSION),
    SiteType("ES", "Estuary", "Estuary", SiteTypeSlug.ESTUARY),
    SiteType("SB-TSM", "Tunl/mine", "Tunnel, shaft, or mine", SiteTypeSlug.TUNNEL),
    SiteType("FA-FON", "Agric area", "Field, Pasture, Orchard, or Nursery", SiteTypeSlug.FIELD),
    SiteType("FA-STS", "Sewer-strm", "Storm sewer", SiteTypeSlug.STORM_SEWER),
    SiteType("FA-CS", "Sewer-comb", "Combined sewer", SiteTypeSlug.COMBINED_SEWER),
    SiteType("FA-OF", "Outfall", "Outfall", SiteTypeSlug.OUTFALL)
]
"""List of USGS site types to retrieve for master site table."""

STATE_LIST: list[us.states.State] = us.states.STATES + [us.states.PR, us.states.DC]
"""List of US states."""

SITE_SCHEMA: pl.Schema = pl.Schema({
    "id": pl.String,
    "vertical_datum": pl.String,
    "original_horizontal_datum_name": pl.String,
    "well_constructed_depth": pl.String,
    "country_name": pl.String,
    "vertical_datum_name": pl.String,
    "drainage_area": pl.Float64,
    "hole_constructed_depth": pl.String,
    "minor_civil_division_code": pl.String,
    "hydrologic_unit_code": pl.String,
    "horizontal_positional_accuracy_code": pl.String,
    "contributing_drainage_area": pl.Float64,
    "depth_source_code": pl.String,
    "agency_name": pl.String,
    "basin_code": pl.String,
    "horizontal_positional_accuracy": pl.String,
    "time_zone_abbreviation": pl.String,
    "altitude": pl.Float64,
    "monitoring_location_name": pl.String,
    "district_code": pl.String,
    "state_code": pl.String,
    "site_type": pl.String,
    "horizontal_position_method_code": pl.String,
    "uses_daylight_savings": pl.String,
    "agency_code": pl.String,
    "country_code": pl.String,
    "county_code": pl.String,
    "altitude_accuracy": pl.Float64,
    "construction_date": pl.String,
    "aquifer_code": pl.String,
    "monitoring_location_number": pl.String,
    "state_name": pl.String,
    "site_type_code": pl.String,
    "altitude_method_code": pl.String,
    "horizontal_position_method_name": pl.String,
    "national_aquifer_code": pl.String,
    "county_name": pl.String,
    "altitude_method_name": pl.String,
    "original_horizontal_datum": pl.String,
    "aquifer_type_code": pl.String,
    "longitude": pl.Float64,
    "latitude": pl.Float64
})
"""Schema for USGS site data."""

class PlotlyFigure(TypedDict):
    """
    Specifies plotly figure dict for use with panel.

    Attributes
    ----
    data: list[plotly.basedatatypes.BaseTraceType]
        List of plotly traces.
    layout: plotly.graph_objects.Layout
        Plotly layout.
    """
    data: list[BaseTraceType]
    layout: go.Layout

DEFAULT_ZOOM: dict[ModelDomainDisplay, int] = {
    ModelDomainDisplay.ALASKA: 5,
    ModelDomainDisplay.CONUS: 3,
    ModelDomainDisplay.HAWAII: 6,
    ModelDomainDisplay.PUERTO_RICO: 8
}
"""Default map zoom for each domain."""

DEFAULT_CENTER: dict[ModelDomainDisplay, dict[str, float]] = {
    ModelDomainDisplay.ALASKA: {"lat": 60.84683, "lon": -149.05659},
    ModelDomainDisplay.CONUS: {"lat": 38.83348, "lon": -93.97612},
    ModelDomainDisplay.HAWAII: {"lat": 21.24988, "lon": -157.59606},
    ModelDomainDisplay.PUERTO_RICO: {"lat": 18.21807, "lon": -66.32802}
}
"""Default map center for each domain."""

METRIC_PLOTTING_LIMITS: dict[Metric, tuple[float, float]] = {
    Metric.RELATIVE_MEAN_BIAS: (-1.0, 1.0),
    Metric.PEARSON_CORRELATION_COEFFICIENT: (-1.0, 1.0),
    Metric.NASH_SUTCLIFFE_EFFICIENCY: (-1.0, 1.0),
    Metric.RELATIVE_MEAN: (0.0, 2.0),
    Metric.RELATIVE_STANDARD_DEVIATION: (0.0, 2.0),
    Metric.RELATIVE_MEDIAN: (0.0, 2.0),
    Metric.RELATIVE_MINIMUM: (0.0, 2.0),
    Metric.RELATIVE_MAXIMUM: (0.0, 2.0),
    Metric.KLING_GUPTA_EFFICIENCY: (-1.0, 1.0)
}
"""Mapping from Metrics to plotting limits (cmin, cmax)."""

CONFIGURATION_LINE_TYPE: dict[ModelConfiguration, str] = {
    ModelConfiguration.ANALYSIS_ASSIM_PUERTO_RICO_NO_DA: "markers"
}
"""Mapping from Model configuration to line type for plotting."""

class AxisType(StrEnum):
    """Plotly axis type."""
    NONE = "-"
    LINEAR = "linear"
    LOG = "log"
    DATE = "date"
    CATEGORY = "category"
    MULTICATEGORY = "multicategory"

MetricFunction = Callable[[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64]
    ], None]
"""Type hint for Numba functions that generate metrics."""

SITE_COLUMN_MAPPING: dict[str, str] = {
    "monitoring_location_name": "Name",
    "monitoring_location_number": "Site code",
    "hydrologic_unit_code": "HUC",
    "site_type": "Site type",
    "drainage_area": "Drainage area (sq.mi.)",
    "contributing_drainage_area": "Contrib. drain. area (sq.mi.)"
}
"""Mapping from site data column to pretty string."""

class MeasurementUnits(StrEnum):
    """Streamflow measurement units."""
    CUBIC_FEET_PER_SECOND = "CFS"
    CUBIC_METERS_PER_SECOND = "CMS"
    CUBIC_FEET_PER_SECOND_PER_SQUARE_MILE = "CFS/sq.mi."
    INCHES_PER_HOUR = "inch/h"
    CUMULATIVE_INCHES_PER_HOUR = "inch/h, accum."

COLUMN_DESCRIPTIONS: dict[str, str] = {
    "label": "Evaluation label",
    "configuration": "National Water Model configuration",
    "nwm_feature_id": "National Water Model channel feature identifier",
    "lead_time_hours_min": "Minimum lead time hours, valid time minus reference time",
    "predicted_cfs_min": "Minimum predicted streamflow in cubic feet per second",
    "predicted_cfs_median": "Median predicted streamflow in cubic feet per second",
    "predicted_cfs_max": "Maximum predicted streamflow in cubic feet per second",
    "predicted_value_time_min": "Earliest prediction valid time",
    "predicted_value_time_max": "Latest prediction valid time",
    "reference_time_min": "Earliest prediction reference time, also called issue time",
    "reference_time_max": "Latest prediction reference time, also called issue time",
    "usgs_site_code": "USGS site code",
    "observed_cfs_min": "Minimum observed streamflow in cubic feet per second",
    "observed_cfs_median": "Median observed streamflow in cubic feet per second",
    "observed_cfs_max": "Maximum observed streamflow in cubic feet per second",
    "observed_value_time_min": "Earliest observation valid time",
    "observed_value_time_max": "Latest observation valid time",
    "sample_size": "Number of prediction-observation pairs",
    "nash_sutcliffe_efficiency_min_point": "Nash-Sutcliffe model efficiency of minimum predicted streamflow, point estimate",
    "nash_sutcliffe_efficiency_min_lower": "Nash-Sutcliffe model efficiency of minimum predicted streamflow, lower 95% confidence estimate",
    "nash_sutcliffe_efficiency_min_upper": "Nash-Sutcliffe model efficiency of minimum predicted streamflow, upper 95% confidence estimate",
    "relative_mean_bias_min_point": "Relative mean bias/error of minimum predicted streamflow, point estimate",
    "relative_mean_bias_min_lower": "Relative mean bias/error of minimum predicted streamflow, lower 95% confidence estimate",
    "relative_mean_bias_min_upper": "Relative mean bias/error of minimum predicted streamflow, upper 95% confidence estimate",
    "pearson_correlation_coefficient_min_point": "Pearson correlation coefficient of minimum predicted streamflow, point estimate",
    "pearson_correlation_coefficient_min_lower": "Pearson correlation coefficient of minimum predicted streamflow, lower 95% confidence estimate",
    "pearson_correlation_coefficient_min_upper": "Pearson correlation coefficient of minimum predicted streamflow, upper 95% confidence estimate",
    "relative_mean_min_point": "Relative mean of minimum predicted streamflow, point estimate",
    "relative_mean_min_lower": "Relative mean of minimum predicted streamflow, lower 95% confidence estimate",
    "relative_mean_min_upper": "Relative mean of minimum predicted streamflow, upper 95% confidence estimate",
    "relative_median_min_point": "Relative median of minimum predicted streamflow, point estimate",
    "relative_median_min_lower": "Relative median of minimum predicted streamflow, lower 95% confidence estimate",
    "relative_median_min_upper": "Relative median of minimum predicted streamflow, upper 95% confidence estimate",
    "relative_minimum_min_point": "Relative minimum of minimum predicted streamflow, point estimate",
    "relative_minimum_min_lower": "Relative minimum of minimum predicted streamflow, lower 95% confidence estimate",
    "relative_minimum_min_upper": "Relative minimum of minimum predicted streamflow, upper 95% confidence estimate",
    "relative_maximum_min_point": "Relative maximum of minimum predicted streamflow, point estimate",
    "relative_maximum_min_lower": "Relative maximum of minimum predicted streamflow, lower 95% confidence estimate",
    "relative_maximum_min_upper": "Relative maximum of minimum predicted streamflow, upper 95% confidence estimate",
    "relative_standard_deviation_min_point": "Relative standard deviation of minimum predicted streamflow, point estimate",
    "relative_standard_deviation_min_lower": "Relative standard deviation of minimum predicted streamflow, lower 95% confidence estimate",
    "relative_standard_deviation_min_upper": "Relative standard deviation of minimum predicted streamflow, upper 95% confidence estimate",
    "kling_gupta_efficiency_min_point": "Kling-Gupta model efficiency of minimum predicted streamflow, point estimate",
    "kling_gupta_efficiency_min_lower": "Kling-Gupta model efficiency of minimum predicted streamflow, lower 95% confidence estimate",
    "kling_gupta_efficiency_min_upper": "Kling-Gupta model efficiency of minimum predicted streamflow, upper 95% confidence estimate",
    "nash_sutcliffe_efficiency_median_point": "Nash-Sutcliffe model efficiency of median predicted streamflow, point estimate",
    "nash_sutcliffe_efficiency_median_lower": "Nash-Sutcliffe model efficiency of median predicted streamflow, lower 95% confidence estimate",
    "nash_sutcliffe_efficiency_median_upper": "Nash-Sutcliffe model efficiency of median predicted streamflow, upper 95% confidence estimate",
    "relative_mean_bias_median_point": "Relative mean bias/error of median predicted streamflow, point estimate",
    "relative_mean_bias_median_lower": "Relative mean bias/error of median predicted streamflow, lower 95% confidence estimate",
    "relative_mean_bias_median_upper": "Relative mean bias/error of median predicted streamflow, upper 95% confidence estimate",
    "pearson_correlation_coefficient_median_point": "Pearson correlation coefficient of median predicted streamflow, point estimate",
    "pearson_correlation_coefficient_median_lower": "Pearson correlation coefficient of median predicted streamflow, lower 95% confidence estimate",
    "pearson_correlation_coefficient_median_upper": "Pearson correlation coefficient of median predicted streamflow, upper 95% confidence estimate",
    "relative_mean_median_point": "Relative mean of median predicted streamflow, point estimate",
    "relative_mean_median_lower": "Relative mean of median predicted streamflow, lower 95% confidence estimate",
    "relative_mean_median_upper": "Relative mean of median predicted streamflow, point estimate",
    "relative_median_median_point": "Relative median of median predicted streamflow, point estimate",
    "relative_median_median_lower": "Relative median of median predicted streamflow, lower 95% confidence estimate",
    "relative_median_median_upper": "Relative median of median predicted streamflow, point estimate",
    "relative_minimum_median_point": "Relative minimum of median predicted streamflow, point estimate",
    "relative_minimum_median_lower": "Relative minimum of median predicted streamflow, lower 95% confidence estimate",
    "relative_minimum_median_upper": "Relative minimum of median predicted streamflow, point estimate",
    "relative_maximum_median_point": "Relative maximum of median predicted streamflow, point estimate",
    "relative_maximum_median_lower": "Relative maximum of median predicted streamflow, lower 95% confidence estimate",
    "relative_maximum_median_upper": "Relative maximum of median predicted streamflow, upper 95% confidence estimate",
    "relative_standard_deviation_median_point": "Relative standard deviation of median predicted streamflow, point estimate",
    "relative_standard_deviation_median_lower": "Relative standard deviation of median predicted streamflow, lower 95% confidence estimate",
    "relative_standard_deviation_median_upper": "Relative standard deviation of median predicted streamflow, upper 95% confidence estimate",
    "kling_gupta_efficiency_median_point": "Kling-Gupta model efficiency of median predicted streamflow, point estimate",
    "kling_gupta_efficiency_median_lower": "Kling-Gupta model efficiency of median predicted streamflow, lower 95% confidence estimate",
    "kling_gupta_efficiency_median_upper": "Kling-Gupta model efficiency of median predicted streamflow, upper 95% confidence estimate",
    "nash_sutcliffe_efficiency_max_point": "Nash-Sutcliffe model efficiency of maximum predicted streamflow, point estimate",
    "nash_sutcliffe_efficiency_max_lower": "Nash-Sutcliffe model efficiency of maximum predicted streamflow, lower 95% confidence estimate",
    "nash_sutcliffe_efficiency_max_upper": "Nash-Sutcliffe model efficiency of maximum predicted streamflow, upper 95% confidence estimate",
    "relative_mean_bias_max_point": "Relative mean bias/error of maximum predicted streamflow, point estimate",
    "relative_mean_bias_max_lower": "Relative mean bias/error of maximum predicted streamflow, lower 95% confidence estimate",
    "relative_mean_bias_max_upper": "Relative mean bias/error of maximum predicted streamflow, upper 95% confidence estimate",
    "pearson_correlation_coefficient_max_point": "Pearson correlation coefficient of maximum predicted streamflow, point estimate",
    "pearson_correlation_coefficient_max_lower": "Pearson correlation coefficient of maximum predicted streamflow, lower 95% confidence estimate",
    "pearson_correlation_coefficient_max_upper": "Pearson correlation coefficient of maximum predicted streamflow, upper 95% confidence estimate",
    "relative_mean_max_point": "Relative mean of maximum predicted streamflow, point estimate",
    "relative_mean_max_lower": "Relative mean of maximum predicted streamflow, lower 95% confidence estimate",
    "relative_mean_max_upper": "Relative mean of maximum predicted streamflow, upper 95% confidence estimate",
    "relative_median_max_point": "Relative median of maximum predicted streamflow, point estimate",
    "relative_median_max_lower": "Relative median of maximum predicted streamflow, lower 95% confidence estimate",
    "relative_median_max_upper": "Relative median of maximum predicted streamflow, upper 95% confidence estimate",
    "relative_minimum_max_point": "Relative minimum of maximum predicted streamflow, point estimate",
    "relative_minimum_max_lower": "Relative minimum of maximum predicted streamflow, lower 95% confidence estimate",
    "relative_minimum_max_upper": "Relative minimum of maximum predicted streamflow, upper 95% confidence estimate",
    "relative_maximum_max_point": "Relative maximum of maximum predicted streamflow, point estimate",
    "relative_maximum_max_lower": "Relative maximum of maximum predicted streamflow, lower 95% confidence estimate",
    "relative_maximum_max_upper": "Relative maximum of maximum predicted streamflow, upper 95% confidence estimate",
    "relative_standard_deviation_max_point": "Relative standard deviation of maximum predicted streamflow, point estimate",
    "relative_standard_deviation_max_lower": "Relative standard deviation of maximum predicted streamflow, lower 95% confidence estimate",
    "relative_standard_deviation_max_upper": "Relative standard deviation of maximum predicted streamflow, upper 95% confidence estimate",
    "kling_gupta_efficiency_max_point": "Kling-Gupta model efficiency of maximum predicted streamflow, point estimate",
    "kling_gupta_efficiency_max_lower": "Kling-Gupta model efficiency of maximum predicted streamflow, lower 95% confidence estimate",
    "kling_gupta_efficiency_max_upper": "Kling-Gupta model efficiency of maximum predicted streamflow, upper 95% confidence estimate"
}
"""Mapping from columns to descriptions."""
