"""Various mappings."""
from dataclasses import dataclass
from enum import StrEnum
from typing import Callable
import polars as pl
import nwm_explorer.urls as nwm_urls

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

@dataclass
class LeadTimeSpec:
    """Dataclass for storing lead time specifications."""
    duration: pl.Duration
    label: str
    sampling_frequency: str

LEAD_TIME_FREQUENCY: dict[Configuration, LeadTimeSpec] = {
    Configuration.medium_range_mem1: LeadTimeSpec(pl.duration(days=1), "lead_time_days_min", "1d"),
    Configuration.medium_range_blend: LeadTimeSpec(pl.duration(days=1), "lead_time_days_min", "1d"),
    Configuration.medium_range_no_da: LeadTimeSpec(pl.duration(days=1), "lead_time_days_min", "1d"),
    Configuration.medium_range_alaska_mem1: LeadTimeSpec(pl.duration(days=1), "lead_time_days_min", "1d"),
    Configuration.medium_range_blend_alaska: LeadTimeSpec(pl.duration(days=1), "lead_time_days_min", "1d"),
    Configuration.medium_range_alaska_no_da: LeadTimeSpec(pl.duration(days=1), "lead_time_days_min", "1d"),
    Configuration.short_range: LeadTimeSpec(pl.duration(hours=6), "lead_time_hours_min", "6h"),
    Configuration.short_range_alaska: LeadTimeSpec(pl.duration(hours=5), "lead_time_hours_min", "5h"),
    Configuration.short_range_hawaii: LeadTimeSpec(pl.duration(hours=6), "lead_time_hours_min", "6h"),
    Configuration.short_range_hawaii_no_da: LeadTimeSpec(pl.duration(hours=6), "lead_time_hours_min", "6h"),
    Configuration.short_range_puertorico: LeadTimeSpec(pl.duration(hours=6), "lead_time_hours_min", "6h"),
    Configuration.short_range_puertorico_no_da: LeadTimeSpec(pl.duration(hours=6), "lead_time_hours_min", "6h")
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
