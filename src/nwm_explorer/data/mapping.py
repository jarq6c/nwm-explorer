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

DEFAULT_ZOOM: dict[ModelDomain, int] = {
    ModelDomain.alaska: 5,
    ModelDomain.conus: 3,
    ModelDomain.hawaii: 6,
    ModelDomain.puertorico: 8
}
"""Default map zoom for each domain."""

DEFAULT_CENTER: dict[ModelDomain, dict[str, float]] = {
    ModelDomain.alaska: {"lat": 60.84683, "lon": -149.05659},
    ModelDomain.conus: {"lat": 38.83348, "lon": -93.97612},
    ModelDomain.hawaii: {"lat": 21.24988, "lon": -157.59606},
    ModelDomain.puertorico: {"lat": 18.21807, "lon": -66.32802}
}
"""Default map center for each domain."""

METRIC_PLOTTING_LIMITS: dict[Metric, tuple[float, float]] = {
    Metric.relative_mean_bias: (-1.0, 1.0),
    Metric.pearson_correlation_coefficient: (-1.0, 1.0),
    Metric.nash_sutcliffe_efficiency: (-1.0, 1.0),
    Metric.relative_mean: (0.0, 2.0),
    Metric.relative_standard_deviation: (0.0, 2.0),
    Metric.kling_gupta_efficiency: (-1.0, 1.0)
}
"""Mapping from Metrics to plotting limits (cmin, cmax)."""