"""Various mappings."""
from dataclasses import dataclass
from enum import StrEnum
from typing import Callable
import polars as pl
import pandas as pd
import nwm_explorer.urls as nwm_urls

EVALUATIONS: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {
    "FY2024_Q1": (pd.Timestamp("2023-10-01"), pd.Timestamp("2024-01-01")),
    "FY2024_Q2": (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-04-01")),
    "FY2024_Q3": (pd.Timestamp("2024-04-01"), pd.Timestamp("2024-07-01")),
    "FY2024_Q4": (pd.Timestamp("2024-07-01"), pd.Timestamp("2024-10-01")),
    "FY2025_Q1": (pd.Timestamp("2024-10-01"), pd.Timestamp("2025-01-01")),
    "FY2025_Q2": (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-04-01")),
    "FY2025_Q3": (pd.Timestamp("2025-04-01"), pd.Timestamp("2025-07-01"))
}
"""Standard evaluations."""

TIMEZONE_MAPPING: dict[str, str] = {
    "AKST": "America/Anchorage",
    "AKDT": "America/Anchorage",
    "HST": "America/Adak",
    "HDT": "America/Adak",
    "AST": "America/Puerto_Rico",
    "CDT": "America/Chicago",
    "CST": "America/Chicago",
    "EDT": "America/New_York",
    "EST": "America/New_York",
    "MST": "America/Phoenix",
    "MDT": "America/Denver",
    "PST": "America/Los_Angeles",
    "PDT": "America/Los_Angeles"
}
"""Mapping from common timezone strings to IANA compatible strings."""

ROUTELINK_SCHEMA: dict[str, pl.DataType] = {
    "nwm_feature_id": pl.Int64,
    "usgs_site_code": pl.String,
    "latitude": pl.Float64,
    "longitude": pl.Float64
}
"""Mapping from routelink field to polars datatype."""

class Domain(StrEnum):
    """Symbols used to reference different model domains."""
    alaska = "alaska"
    conus = "conus"
    hawaii = "hawaii"
    puertorico = "puertorico"

DOMAIN_MAPPING: dict[str, Domain] = {
    "alaska": Domain.alaska,
    "conus": Domain.conus,
    "hawaii": Domain.hawaii,
    "puertorico": Domain.puertorico,
    "RouteLink_AK.csv": Domain.alaska,
    "RouteLink_CONUS.csv": Domain.conus,
    "RouteLink_HI.csv": Domain.hawaii,
    "RouteLink_PRVI.csv": Domain.puertorico,
    "Alaska": Domain.alaska,
    "CONUS": Domain.conus,
    "Hawaii": Domain.hawaii,
    "Puerto Rico": Domain.puertorico
}
"""Mapping from common strings to standard symbols."""

class Configuration(StrEnum):
    """Symbols used to reference data configurations."""
    analysis_assim_extend_alaska_no_da = "analysis_assim_extend_alaska_no_da"
    analysis_assim_extend_no_da = "analysis_assim_extend_no_da"
    analysis_assim_hawaii_no_da = "analysis_assim_hawaii_no_da"
    analysis_assim_puertorico_no_da = "analysis_assim_puertorico_no_da"
    medium_range_mem1 = "medium_range_mem1"
    medium_range_blend = "medium_range_blend"
    medium_range_no_da = "medium_range_no_da"
    medium_range_alaska_mem1 = "medium_range_alaska_mem1"
    medium_range_blend_alaska = "medium_range_blend_alaska"
    medium_range_alaska_no_da = "medium_range_alaska_no_da"
    short_range = "short_range"
    short_range_alaska = "short_range_alaska"
    short_range_hawaii = "short_range_hawaii"
    short_range_hawaii_no_da = "short_range_hawaii_no_da"
    short_range_puertorico = "short_range_puertorico"
    short_range_puertorico_no_da = "short_range_puertorico_no_da"
    usgs = "usgs"

LEAD_TIME_FREQUENCY: dict[Configuration, pl.Duration] = {
    Configuration.medium_range_mem1: (pl.duration(hours=24), "1d"),
    Configuration.medium_range_blend: (pl.duration(hours=24), "1d"),
    Configuration.medium_range_no_da: (pl.duration(hours=24), "1d"),
    Configuration.medium_range_alaska_mem1: (pl.duration(hours=24), "1d"),
    Configuration.medium_range_blend_alaska: (pl.duration(hours=24), "1d"),
    Configuration.medium_range_alaska_no_da: (pl.duration(hours=24), "1d"),
    Configuration.short_range: (pl.duration(hours=6), "6h"),
    Configuration.short_range_alaska: (pl.duration(hours=5), "5h"),
    Configuration.short_range_hawaii: (pl.duration(hours=6), "6h"),
    Configuration.short_range_hawaii_no_da: (pl.duration(hours=6), "6h"),
    Configuration.short_range_puertorico: (pl.duration(hours=6), "6h"),
    Configuration.short_range_puertorico_no_da: (pl.duration(hours=6), "6h")
}
"""Mapping used for computing lead time and sampling frequency."""

class FileType(StrEnum):
    """Symbols used for common file types."""
    netcdf = "netcdf"
    parquet = "parquet"
    tsv = "tsv"

class Variable(StrEnum):
    """Symbols used for common variables."""
    streamflow = "streamflow"
    streamflow_pairs = "streamflow_pairs"
    streamflow_metrics = "streamflow_metrics"

class Units(StrEnum):
    """Symbols used for common units."""
    cubic_feet_per_second = "cfs"
    metrics = "metrics"

class Confidence(StrEnum):
    """Symbols used to describe confidence interval range estimates."""
    point = "point"
    lower = "lower"
    upper = "upper"

DEFAULT_ZOOM: dict[Domain, int] = {
    Domain.alaska: 5,
    Domain.conus: 3,
    Domain.hawaii: 6,
    Domain.puertorico: 8
}
"""Default map zoom for each domain."""

NWM_URL_BUILDERS: dict[tuple[Domain, Configuration], Callable] = {
    (Domain.alaska, Configuration.analysis_assim_extend_alaska_no_da): nwm_urls.analysis_assim_extend_alaska_no_da,
    (Domain.conus, Configuration.analysis_assim_extend_no_da): nwm_urls.analysis_assim_extend_no_da,
    (Domain.hawaii, Configuration.analysis_assim_hawaii_no_da): nwm_urls.analysis_assim_hawaii_no_da,
    (Domain.puertorico, Configuration.analysis_assim_puertorico_no_da): nwm_urls.analysis_assim_puertorico_no_da,
    (Domain.conus, Configuration.medium_range_mem1): nwm_urls.medium_range_mem1,
    (Domain.conus, Configuration.medium_range_blend): nwm_urls.medium_range_blend,
    (Domain.conus, Configuration.medium_range_no_da): nwm_urls.medium_range_no_da,
    (Domain.alaska, Configuration.medium_range_alaska_mem1): nwm_urls.medium_range_alaska_mem1,
    (Domain.alaska, Configuration.medium_range_blend_alaska): nwm_urls.medium_range_blend_alaska,
    (Domain.alaska, Configuration.medium_range_alaska_no_da): nwm_urls.medium_range_alaska_no_da,
    (Domain.conus, Configuration.short_range): nwm_urls.short_range,
    (Domain.alaska, Configuration.short_range_alaska): nwm_urls.short_range_alaska,
    (Domain.hawaii, Configuration.short_range_hawaii): nwm_urls.short_range_hawaii,
    (Domain.hawaii, Configuration.short_range_hawaii_no_da): nwm_urls.short_range_hawaii_no_da,
    (Domain.puertorico, Configuration.short_range_puertorico): nwm_urls.short_range_puertorico,
    (Domain.puertorico, Configuration.short_range_puertorico_no_da): nwm_urls.short_range_puertorico_no_da
}
"""Mapping from (Domain, Configuration) to url builder function."""

DOMAIN_CONFIGURATION_MAPPING: dict[Domain, dict[str, Configuration]] = {
    Domain.alaska: {
        "Extended Analysis (MRMS, No-DA)": Configuration.analysis_assim_extend_alaska_no_da,
        "Medium Range Forecast (GFS, Deterministic)": Configuration.medium_range_alaska_mem1,
        "Medium Range Forecast (NBM, Deterministic)": Configuration.medium_range_blend_alaska,
        "Medium Range Forecast (GFS, Deterministic, No-DA)": Configuration.medium_range_alaska_no_da,
        "Short Range Forecast (HRRR)": Configuration.short_range_alaska
    },
    Domain.conus: {
        "Extended Analysis (MRMS, No-DA)": Configuration.analysis_assim_extend_no_da,
        "Medium Range Forecast (GFS, Deterministic)": Configuration.medium_range_mem1,
        "Medium Range Forecast (NBM, Deterministic)": Configuration.medium_range_blend,
        "Medium Range Forecast (GFS, Deterministic, No-DA)": Configuration.medium_range_no_da,
        "Short Range Forecast (HRRR)": Configuration.short_range
    },
    Domain.hawaii: {
        "Analysis (MRMS, No-DA)": Configuration.analysis_assim_hawaii_no_da,
        "Short Range Forecast (WRF-ARW)": Configuration.short_range_hawaii,
        "Short Range Forecast (WRF-ARW, No-DA)": Configuration.short_range_hawaii_no_da
    },
    Domain.puertorico: {
        "Analysis (MRMS, No-DA)": Configuration.analysis_assim_puertorico_no_da,
        "Short Range Forecast (WRF-ARW)": Configuration.short_range_puertorico,
        "Short Range Forecast (WRF-ARW, No-DA)": Configuration.short_range_puertorico_no_da
    }
}
"""Mapping from domains to pretty string representations of model configurations."""

LEAD_TIME_VALUES: dict[Configuration, list[int]] = {
    Configuration.medium_range_mem1: [l for l in range(0, 240, 24)],
    Configuration.medium_range_blend: [l for l in range(0, 240, 24)],
    Configuration.medium_range_no_da: [l for l in range(0, 240, 24)],
    Configuration.medium_range_alaska_mem1: [l for l in range(0, 240, 24)],
    Configuration.medium_range_blend_alaska: [l for l in range(0, 240, 24)],
    Configuration.medium_range_alaska_no_da: [l for l in range(0, 240, 24)],
    Configuration.short_range: [l for l in range(0, 18, 6)],
    Configuration.short_range_alaska: [l for l in range(0, 45, 5)],
    Configuration.short_range_hawaii: [l for l in range(0, 48, 6)],
    Configuration.short_range_hawaii_no_da: [l for l in range(0, 48, 6)],
    Configuration.short_range_puertorico: [l for l in range(0, 48, 6)],
    Configuration.short_range_puertorico_no_da: [l for l in range(0, 48, 6)]
}
"""Mapping used to display available lead times."""

DOMAIN_STRINGS: dict[str, Domain] = {
    "Alaska": DOMAIN_MAPPING["Alaska"],
    "CONUS": DOMAIN_MAPPING["CONUS"],
    "Hawaii": DOMAIN_MAPPING["Hawaii"],
    "Puerto Rico": DOMAIN_MAPPING["Puerto Rico"]
}
"""Mapping from pretty strings to domains for display."""

CONFIDENCE_STRINGS: dict[str, Confidence] = {
    "Point": Confidence.point,
    "Lower": Confidence.lower,
    "Upper": Confidence.upper
}
"""Mapping from pretty strings to confidence boundaries for display."""

class Metric(StrEnum):
    """Symbols used to reference different evaluation metrics."""
    nash_sutcliffe_efficiency = "nash_sutcliffe_efficiency"
    mean_relative_bias = "mean_relative_bias"
    pearson_correlation_coefficient = "pearson_correlation_coefficient"
    relative_variability = "relative_variability"
    relative_mean = "relative_mean"
    kling_gupta_efficiency = "kling_gupta_efficiency"

METRIC_STRINGS: dict[str, Metric] = {
    "Mean relative bias": Metric.mean_relative_bias,
    "Pearson correlation coefficient": Metric.pearson_correlation_coefficient,
    "Nash-Sutcliffe efficiency": Metric.nash_sutcliffe_efficiency,
    "Relative mean": Metric.relative_mean,
    "Relative variability": Metric.relative_variability,
    "Kling-Gupta efficiency": Metric.kling_gupta_efficiency
}
"""Mapping from pretty strings to evaluation metrics for display."""

METRIC_SHORTHAND: dict[Metric, str] = {
    Metric.mean_relative_bias: "rmb",
    Metric.pearson_correlation_coefficient: "pearson",
    Metric.nash_sutcliffe_efficiency: "nse",
    Metric.relative_mean: "rel_mean",
    Metric.relative_variability: "rel_var",
    Metric.kling_gupta_efficiency: "kge"
}
"""Mapping from Metrics to short column names."""

METRIC_PLOTTING_LIMITS: dict[Metric, tuple[float, float]] = {
    Metric.mean_relative_bias: (-1.0, 1.0),
    Metric.pearson_correlation_coefficient: (-1.0, 1.0),
    Metric.nash_sutcliffe_efficiency: (-1.0, 1.0),
    Metric.relative_mean: (0.0, 2.0),
    Metric.relative_variability: (0.0, 2.0),
    Metric.kling_gupta_efficiency: (-1.0, 1.0)
}
"""Mapping from Metrics to plotting limist (cmin, cmax)."""

CONFIDENCE_SHORTHAND: dict[Confidence, str] = {
    Confidence.point: "",
    Confidence.lower: "_lower",
    Confidence.upper: "_upper"
}
"""Mapping from pretty strings to confidence boundaries for display."""

@dataclass
class HistogramSpec:
    """Histogram parameters."""
    bin_width: float
    xmin: float
    xmax: float

METRIC_HISTOGRAMS: dict[Metric, HistogramSpec] = {
    Metric.nash_sutcliffe_efficiency: HistogramSpec(0.2, -1.0, 1.0),
    Metric.mean_relative_bias: HistogramSpec(0.2, -1.0, 1.0),
    Metric.pearson_correlation_coefficient: HistogramSpec(0.2, -1.0, 1.0),
    Metric.relative_variability: HistogramSpec(0.2, -1.0, 1.0),
    Metric.relative_mean: HistogramSpec(0.2, -1.0, 1.0),
    Metric.kling_gupta_efficiency: HistogramSpec(0.2, -1.0, 1.0)
}
"""Mapping from Metrics to histogram parameters."""

class CallbackType(StrEnum):
    """Symbols used to reference different types of callbacks."""
    evaluation = "evaluation"
    domain = "domain"
    configuration = "configuration"
    threshold = "threshold"
    metric = "metric"
    confidence = "confidence"
    lead_time = "lead_time"
    click = "click"
    relayout = "relayout"
