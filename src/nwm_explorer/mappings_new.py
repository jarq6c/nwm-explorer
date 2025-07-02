"""Enums and common mappings."""
from enum import StrEnum
import pandas as pd

class Evaluation(StrEnum):
    """Standard evaluations enums."""
    fy2024_q1 = "fy2024_q1"
    fy2024_q2 = "fy2024_q2"
    fy2024_q3 = "fy2024_q3"
    fy2024_q4 = "fy2024_q4"
    fy2025_q1 = "fy2025_q1"
    fy2025_q2 = "fy2025_q2"
    fy2025_q3 = "fy2025_q3"
    all_data = "all_data"

EVALUATION_PERIODS: dict[Evaluation, tuple[pd.Timestamp, pd.Timestamp]] = {
    Evaluation.fy2024_q1: (pd.Timestamp("2023-10-01"), pd.Timestamp("2024-01-01")),
    Evaluation.fy2024_q2: (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-04-01")),
    Evaluation.fy2024_q3: (pd.Timestamp("2024-04-01"), pd.Timestamp("2024-07-01")),
    Evaluation.fy2024_q4: (pd.Timestamp("2024-07-01"), pd.Timestamp("2024-10-01")),
    Evaluation.fy2025_q1: (pd.Timestamp("2024-10-01"), pd.Timestamp("2025-01-01")),
    Evaluation.fy2025_q2: (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-04-01")),
    Evaluation.fy2025_q3: (pd.Timestamp("2025-04-01"), pd.Timestamp("2025-07-01")),
    Evaluation.all_data: (pd.Timestamp("2023-10-01"), pd.Timestamp("2025-07-01"))
}
"""Mapping from Evaluation enums to evaluation periods."""

EVALUATION_STRINGS: dict[str, Evaluation] = {
    "FY2024 Q1": Evaluation.fy2024_q1,
    "FY2024 Q2": Evaluation.fy2024_q2,
    "FY2024 Q3": Evaluation.fy2024_q3,
    "FY2024 Q4": Evaluation.fy2024_q4,
    "FY2025 Q1": Evaluation.fy2025_q1,
    "FY2025 Q1": Evaluation.fy2025_q2,
    "FY2025 Q1": Evaluation.fy2025_q3,
    "All data": Evaluation.all_data
}
"""Mapping from pretty evaluation strings to Evaluation enums."""

class Domain(StrEnum):
    """Model domain enums."""
    alaska = "alaska"
    conus = "conus"
    hawaii = "hawaii"
    puertorico = "puertorico"

DOMAIN_STRINGS: dict[str, Evaluation] = {
    "Alaska": Domain.alaska,
    "CONUS": Domain.conus,
    "Hawaii": Domain.hawaii,
    "Puerto Rico": Domain.puertorico
}
"""Mapping from pretty domain strings to Domain enums."""

class Configuration(StrEnum):
    """Model configuration enums."""
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

DOMAIN_CONFIGURATIONS: dict[Domain, dict[str, Configuration]] = {
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
"""
Mapping from domains to pretty string representations of model configurations.
Pretty strings map to model Configuration enums.
"""

class Metric(StrEnum):
    """Model evaluation metric enums."""
    nash_sutcliffe_efficiency = "nash_sutcliffe_efficiency"
    mean_relative_bias = "mean_relative_bias"
    pearson_correlation_coefficient = "pearson_correlation_coefficient"
    relative_standard_deviation = "relative_standard_deviation"
    relative_mean = "relative_mean"
    kling_gupta_efficiency = "kling_gupta_efficiency"
    coefficient_of_persistence = "coefficient_of_persistence"
    coefficient_of_extrapolation = "coefficient_of_extrapolation"

METRIC_STRINGS: dict[str, Metric] = {
    "Mean relative bias": Metric.mean_relative_bias,
    "Pearson correlation coefficient": Metric.pearson_correlation_coefficient,
    "Nash-Sutcliffe efficiency": Metric.nash_sutcliffe_efficiency,
    "Relative mean": Metric.relative_mean,
    "Relative standard deviation": Metric.relative_standard_deviation,
    "Kling-Gupta efficiency": Metric.kling_gupta_efficiency,
    "Coefficient of persistence": Metric.coefficient_of_persistence,
    "Coefficient of extrapolation": Metric.coefficient_of_extrapolation
}
"""Mapping from pretty strings to model performance Metric enums."""

class Confidence(StrEnum):
    """Confidence interval enums."""
    point = "point"
    lower = "lower"
    upper = "upper"

CONFIDENCE_STRINGS: dict[str, Confidence] = {
    "Point": Confidence.point,
    "Lower": Confidence.lower,
    "Upper": Confidence.upper
}
"""Mapping from pretty strings to Confidence enums."""

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
"""Mapping from model Configuration enums to lists of lead time integers (hours)."""

class CallbackType(StrEnum):
    """Callback type enums."""
    evaluation = "evaluation"
    domain = "domain"
    configuration = "configuration"
    threshold = "threshold"
    metric = "metric"
    confidence = "confidence"
    lead_time = "lead_time"
    click = "click"
    relayout = "relayout"
