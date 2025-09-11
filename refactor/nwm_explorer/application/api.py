"""
Objects and methods needed to support programmatic interaction with various
application components.
"""
from pathlib import Path
from enum import StrEnum
from pydantic import BaseModel
import polars as pl

class ModelDomain(StrEnum):
    """Symbols that refer to model spatial domains."""
    conus = "CONUS"
    alaska = "Alaska"
    hawaii = "Hawaii"
    puertorico = "Puerto Rico"

class ModelForcing(StrEnum):
    """Symbols that refer to model forcing."""
    eana_mrms = "Extended Analysis (MRMS, No-DA)"
    mrf_gfs = "Medium Range Forecast (GFS, Deterministic)"
    mrf_nbm = "Medium Range Forecast (NBM, Deterministic)"
    mrf_gfs_no_da = "Medium Range Forecast (GFS, Deterministic, No-DA)"
    srf_hrrr = "Short Range Forecast (HRRR)"
    ana_mrms = "Analysis (MRMS, No-DA)"
    srf_wrf = "Short Range Forecast (WRF-ARW)"
    srf_wrf_no_da = "Short Range Forecast (WRF-ARW, No-DA)"

class ModelConfiguration(StrEnum):
    """Symbols that refer to model configurations."""
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

class Threshold(StrEnum):
    """Symbols that refer to different conditional flow thresholds."""
    usgs_aep_100 = "100% AEP-USGS (All data)"
    usgs_aep_20 = "20% AEP-USGS"
    usgs_aep_02 = "2% AEP-USGS"
    usgs_aep_01 = "1% AEP-USGS"
    usgs_aep_001 = "0.1% AEP-USGS"
    usgs_aep_0001 = "0.01% AEP-USGS"
    action = "NWS Action Stage"
    minor = "NWS Minor Stage"
    moderate = "NWS Moderate Stage"
    major = "NWS Major Stage"
    record = "NWS Record Stage"
    high_water = "NWMv3.0 High Water"

class Metric(StrEnum):
    """Symbols that refer to model evaluation metrics."""
    nash_sutcliffe_efficiency = "Nash-Sutcliffe Model Efficiency"
    relative_mean_bias = "Relative mean bias"
    pearson_correlation_coefficient = "Pearson Correlation Coefficient"
    relative_mean = "Relative mean"
    relative_standard_deviation = "Relative standard deviation"
    kling_gupta_efficiency = "Kling-Gupta Model Efficiency"

class Confidence(StrEnum):
    """Symbols that refer to condifence estimates for evaluation metrics."""
    _point = "Point"
    _lower = "Lower"
    _upper = "Upper"

DOMAIN_FORCING_CONFIGURATION: dict[ModelDomain, dict[ModelForcing, ModelConfiguration]] = {
    ModelDomain.alaska: {
        ModelForcing.eana_mrms: ModelConfiguration.analysis_assim_extend_alaska_no_da,
        ModelForcing.mrf_gfs: ModelConfiguration.medium_range_alaska_mem1,
        ModelForcing.mrf_nbm: ModelConfiguration.medium_range_blend_alaska,
        ModelForcing.mrf_gfs_no_da: ModelConfiguration.medium_range_alaska_no_da,
        ModelForcing.srf_hrrr: ModelConfiguration.short_range_alaska
    },
    ModelDomain.conus: {
        ModelForcing.eana_mrms: ModelConfiguration.analysis_assim_extend_no_da,
        ModelForcing.mrf_gfs: ModelConfiguration.medium_range_mem1,
        ModelForcing.mrf_nbm: ModelConfiguration.medium_range_blend,
        ModelForcing.mrf_gfs_no_da: ModelConfiguration.medium_range_no_da,
        ModelForcing.srf_hrrr: ModelConfiguration.short_range
    },
    ModelDomain.hawaii: {
        ModelForcing.ana_mrms: ModelConfiguration.analysis_assim_hawaii_no_da,
        ModelForcing.srf_wrf: ModelConfiguration.short_range_hawaii,
        ModelForcing.srf_wrf_no_da: ModelConfiguration.short_range_hawaii_no_da
    },
    ModelDomain.puertorico: {
        ModelForcing.ana_mrms: ModelConfiguration.analysis_assim_puertorico_no_da,
        ModelForcing.srf_wrf: ModelConfiguration.short_range_puertorico,
        ModelForcing.srf_wrf_no_da: ModelConfiguration.short_range_puertorico_no_da
    }
}
"""
Mapping that specifies relationships between ModelDomain, ModelForcing, and
ModelConfiguration.
"""

PREDICTION_SAMPLING: dict[ModelConfiguration, tuple[pl.Duration, str]] = {
    ModelConfiguration.medium_range_mem1: (pl.duration(hours=24), "1d"),
    ModelConfiguration.medium_range_blend: (pl.duration(hours=24), "1d"),
    ModelConfiguration.medium_range_no_da: (pl.duration(hours=24), "1d"),
    ModelConfiguration.medium_range_alaska_mem1: (pl.duration(hours=24), "1d"),
    ModelConfiguration.medium_range_blend_alaska: (pl.duration(hours=24), "1d"),
    ModelConfiguration.medium_range_alaska_no_da: (pl.duration(hours=24), "1d"),
    ModelConfiguration.short_range: (pl.duration(hours=6), "6h"),
    ModelConfiguration.short_range_alaska: (pl.duration(hours=5), "5h"),
    ModelConfiguration.short_range_hawaii: (pl.duration(hours=6), "6h"),
    ModelConfiguration.short_range_hawaii_no_da: (pl.duration(hours=6), "6h"),
    ModelConfiguration.short_range_puertorico: (pl.duration(hours=6), "6h"),
    ModelConfiguration.short_range_puertorico_no_da: (pl.duration(hours=6), "6h")
}
"""Mapping used for computing lead time and sampling frequency."""

LEAD_TIME_VALUES: dict[ModelConfiguration, list[int]] = {
    ModelConfiguration.medium_range_mem1: [l for l in range(0, 240, 24)],
    ModelConfiguration.medium_range_blend: [l for l in range(0, 240, 24)],
    ModelConfiguration.medium_range_no_da: [l for l in range(0, 240, 24)],
    ModelConfiguration.medium_range_alaska_mem1: [l for l in range(0, 240, 24)],
    ModelConfiguration.medium_range_blend_alaska: [l for l in range(0, 240, 24)],
    ModelConfiguration.medium_range_alaska_no_da: [l for l in range(0, 240, 24)],
    ModelConfiguration.short_range: [l for l in range(0, 18, 6)],
    ModelConfiguration.short_range_alaska: [l for l in range(0, 45, 5)],
    ModelConfiguration.short_range_hawaii: [l for l in range(0, 48, 6)],
    ModelConfiguration.short_range_hawaii_no_da: [l for l in range(0, 48, 6)],
    ModelConfiguration.short_range_puertorico: [l for l in range(0, 48, 6)],
    ModelConfiguration.short_range_puertorico_no_da: [l for l in range(0, 48, 6)]
}
"""Mapping from model ModelConfiguration enums to lists of lead time integers (hours)."""

class DashboardConfiguration(BaseModel):
    """
    Dashboard configuration options.

    Attributes
    ----------
    title: str
        Dashboard title displayed on header.
    slug: str
        Dashboard slug appended to hostname in the URL.
    """
    title: str
    slug: str

class EvaluationRegistry(BaseModel):
    """
    Centralized object used to track various options used through the application.

    Attributes
    ----------
    dashboard_configuration: DashboardConfiguration
        Dashboard configuration options.
    evaluations: dict[str, dict[ModelDomain, dict[ModelForcing, Path]]]
        Dict of EvaluationSpec keyed to hashable str.
    """
    dashboard_configuration: DashboardConfiguration
    evaluations: dict[str, dict[ModelDomain, dict[ModelForcing, Path]]]
