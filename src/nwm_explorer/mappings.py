"""Various mappings."""
from dataclasses import dataclass
from enum import StrEnum
import polars as pl

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
    ALASKA = "alaska"
    CONUS = "conus"
    HAWAII = "hawaii"
    PUERTORICO = "puertorico"

DOMAIN_MAPPING: dict[str, Domain] = {
    "alaska": Domain.ALASKA,
    "conus": Domain.CONUS,
    "hawaii": Domain.HAWAII,
    "puertorico": Domain.PUERTORICO,
    "RouteLink_AK.csv": Domain.ALASKA,
    "RouteLink_CONUS.csv": Domain.CONUS,
    "RouteLink_HI.csv": Domain.HAWAII,
    "RouteLink_PRVI.csv": Domain.PUERTORICO
}
"""Mapping from common strings to standard symbols."""

class Configuration(StrEnum):
    """Symbols used to reference data configurations."""
    ANALYSIS = "analysis"
    OBSERVATIONS = "observations"
    MRF_GFS = "mrf_gfs"
    MRF_NBM = "mrf_nbm"

@dataclass
class LeadTimeSpec:
    """Dataclass for storing lead time specifications."""
    duration: pl.Duration
    label: str

LEAD_TIME_FREQUENCY: dict[Configuration, LeadTimeSpec] = {
    Configuration.MRF_GFS: LeadTimeSpec(pl.duration(days=1), "lead_time_days_min"),
    Configuration.MRF_NBM: LeadTimeSpec(pl.duration(days=1), "lead_time_days_min")
}
"""Mapping used for computing lead time."""

class FileType(StrEnum):
    """Symbols used for common file types."""
    NETCDF = "netcdf"
    PARQUET = "parquet"
    TSV = "tsv"

class Variable(StrEnum):
    """Symbols used for common variables."""
    STREAMFLOW = "streamflow"
    STREAMFLOW_PAIRS = "streamflow_pairs"
    STREAMFLOW_METRICS = "streamflow_metrics"

class Units(StrEnum):
    """Symbols used for common units."""
    CUBIC_FEET_PER_SECOND = "cfs"
    METRICS = "metrics"

class Confidence(StrEnum):
    """Symbols used to describe confidence interval range estimates."""
    POINT = "point"
    LOWER = "lower"
    UPPER = "upper"
