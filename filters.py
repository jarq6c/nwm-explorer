"""Filtering widgets."""
from dataclasses import dataclass
from typing import Callable
from enum import StrEnum
import pandas as pd
import panel as pn

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

@dataclass
class FilterState:
    """Dashboard state variables."""
    evaluation: Evaluation
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    domain: Domain
    configuration: Configuration
    threshold: str
    metric: Metric
    metric_label: str
    confidence: Confidence
    lead_time: int

class FilteringWidgets:
    def __init__(self):
        # Filtering options
        self.callbacks: list[Callable] = []
        self.evaluation_filter = pn.widgets.Select(
            name="Evaluation",
            options=list(EVALUATION_STRINGS.keys())
        )
        self.domain_filter = pn.widgets.Select(
            name="Model Domain",
            options=list(DOMAIN_STRINGS.keys())
        )
        domain = DOMAIN_STRINGS[self.domain_filter.value]
        self.configuration_filter = pn.widgets.Select(
            name="Model Configuration",
            options=list(
            DOMAIN_CONFIGURATIONS[domain].keys()
            ))
        self.threshold_filter = pn.widgets.Select(
            name="Streamflow Threshold (≥)",
            options=[
                "100% AEP-USGS (All data)"
            ]
        )
        self.metric_filter = pn.widgets.Select(
            name="Evaluation Metric",
            options=list(METRIC_STRINGS.keys())
        )
        self.confidence_filter = pn.widgets.Select(
            name="Confidence Estimate (95%)",
            options=list(CONFIDENCE_STRINGS.keys())
        )
        configuration = DOMAIN_CONFIGURATIONS[domain][self.configuration_filter.value]
        if configuration in LEAD_TIME_VALUES:
            lead_time_options = LEAD_TIME_VALUES[configuration]
        else:
            lead_time_options = [0]
        self.lead_time_filter = pn.Row(pn.widgets.DiscretePlayer(
            name="Minimum lead time (hours)",
            options=lead_time_options,
            show_loop_controls=False,
            visible_buttons=["previous", "next"],
            width=300
            ))

        def update_configurations(domain_string):
            if domain_string is None:
                return
            domain = DOMAIN_STRINGS[domain_string]
            self.configuration_filter.options = list(
                DOMAIN_CONFIGURATIONS[domain].keys()
            )
        pn.bind(update_configurations, self.domain_filter, watch=True)

        def update_lead_times(event):
            if event is None:
                return
            domain = DOMAIN_STRINGS[self.domain_filter.value]
            configuration = DOMAIN_CONFIGURATIONS[domain][self.configuration_filter.value]
            if configuration in LEAD_TIME_VALUES:
                lead_time_options = LEAD_TIME_VALUES[configuration]
            else:
                lead_time_options = [0]
            v = self.lead_time_filter[0].value
            self.lead_time_filter.objects = [pn.widgets.DiscretePlayer(
                name="Minimum lead time (hours)",
                options=lead_time_options,
                show_loop_controls=False,
                visible_buttons=["previous", "next"],
                width=300,
                value=v if v in lead_time_options else 0
                )]
            for func in self.callbacks:
                pn.bind(func, self.lead_time_filter[0],
                    callback_type=CallbackType.lead_time, watch=True)
        pn.bind(update_lead_times, self.configuration_filter,
            watch=True)
        pn.bind(update_lead_times, self.domain_filter,
            watch=True)

    @property
    def state(self) -> FilterState:
        """Returns current state of filtering options."""
        evaluation = EVALUATION_STRINGS[self.evaluation_filter.value]
        domain = DOMAIN_STRINGS[self.domain_filter.value]
        return FilterState(
            evaluation=evaluation,
            start_date=EVALUATION_PERIODS[evaluation][0],
            end_date=EVALUATION_PERIODS[evaluation][1],
            domain=domain,
            configuration=DOMAIN_CONFIGURATIONS[domain][self.configuration_filter.value],
            threshold=self.threshold_filter.value,
            metric=METRIC_STRINGS[self.metric_filter.value],
            metric_label=self.metric_filter.value,
            confidence=CONFIDENCE_STRINGS[self.confidence_filter.value],
            lead_time=self.lead_time_filter[0].value
        )

    def servable(self) -> pn.Card:
        return pn.Card(pn.Column(
            self.evaluation_filter,
            self.domain_filter,
            self.configuration_filter,
            self.threshold_filter,
            self.metric_filter,
            self.confidence_filter,
            self.lead_time_filter
            ),
            title="Filters",
            collapsible=False
        )

    def register_callback(self, func: Callable) -> None:
        """Register callback function."""
        pn.bind(func, self.evaluation_filter, callback_type=CallbackType.evaluation, watch=True)
        pn.bind(func, self.domain_filter, callback_type=CallbackType.domain, watch=True)
        pn.bind(func, self.configuration_filter, callback_type=CallbackType.configuration, watch=True)
        pn.bind(func, self.threshold_filter, callback_type=CallbackType.threshold, watch=True)
        pn.bind(func, self.metric_filter, callback_type=CallbackType.metric, watch=True)
        pn.bind(func, self.confidence_filter, callback_type=CallbackType.confidence, watch=True)
        pn.bind(func, self.lead_time_filter[0], callback_type=CallbackType.lead_time, watch=True)
        self.callbacks.append(func)

def main():
    filters = FilteringWidgets()
    pn.serve(filters.servable())

if __name__ == "__main__":
    main()
