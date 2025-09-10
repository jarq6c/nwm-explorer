"""
Objects and methods needed to support programmatic interaction with various
application components.
"""
from pathlib import Path
from enum import StrEnum
from pydantic import BaseModel

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
