from enum import StrEnum

class ModelDomain(StrEnum):
    """Model domains."""
    alaska = "alaska"
    hawaii = "hawaii"
    conus = "conus"
    puertorico = "puertorico"

class ModelConfiguration(StrEnum):
    """Data types."""
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

class Metric(StrEnum):
    """Metrics."""
    nash_sutcliffe_efficiency = "nash_sutcliffe_efficiency"
    relative_mean_bias = "relative_mean_bias"
    pearson_correlation_coefficient = "pearson_correlation_coefficient"
    relative_mean = "relative_mean"
    relative_standard_deviation = "relative_standard_deviation"
    kling_gupta_efficiency = "kling_gupta_efficiency"
