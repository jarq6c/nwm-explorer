"""Filtering widgets."""
from enum import StrEnum
from dataclasses import dataclass
from typing import Callable
import panel as pn

from nwm_explorer.data.mapping import ModelDomain, ModelConfiguration, Metric
from nwm_explorer.evaluation.compute import EvaluationRegistry

DOMAIN_STRINGS: dict[str, ModelDomain] = {
    "Alaska": ModelDomain.alaska,
    "CONUS": ModelDomain.conus,
    "Hawaii": ModelDomain.hawaii,
    "Puerto Rico": ModelDomain.puertorico
}
"""Mapping from pretty domain strings to ModelDomain enums."""

DOMAIN_CONFIGURATIONS: dict[ModelDomain, dict[str, ModelConfiguration]] = {
    ModelDomain.alaska: {
        "Extended Analysis (MRMS, No-DA)": ModelConfiguration.analysis_assim_extend_alaska_no_da,
        "Medium Range Forecast (GFS, Deterministic)": ModelConfiguration.medium_range_alaska_mem1,
        "Medium Range Forecast (NBM, Deterministic)": ModelConfiguration.medium_range_blend_alaska,
        "Medium Range Forecast (GFS, Deterministic, No-DA)": ModelConfiguration.medium_range_alaska_no_da,
        "Short Range Forecast (HRRR)": ModelConfiguration.short_range_alaska
    },
    ModelDomain.conus: {
        "Extended Analysis (MRMS, No-DA)": ModelConfiguration.analysis_assim_extend_no_da,
        "Medium Range Forecast (GFS, Deterministic)": ModelConfiguration.medium_range_mem1,
        "Medium Range Forecast (NBM, Deterministic)": ModelConfiguration.medium_range_blend,
        "Medium Range Forecast (GFS, Deterministic, No-DA)": ModelConfiguration.medium_range_no_da,
        "Short Range Forecast (HRRR)": ModelConfiguration.short_range
    },
    ModelDomain.hawaii: {
        "Analysis (MRMS, No-DA)": ModelConfiguration.analysis_assim_hawaii_no_da,
        "Short Range Forecast (WRF-ARW)": ModelConfiguration.short_range_hawaii,
        "Short Range Forecast (WRF-ARW, No-DA)": ModelConfiguration.short_range_hawaii_no_da
    },
    ModelDomain.puertorico: {
        "Analysis (MRMS, No-DA)": ModelConfiguration.analysis_assim_puertorico_no_da,
        "Short Range Forecast (WRF-ARW)": ModelConfiguration.short_range_puertorico,
        "Short Range Forecast (WRF-ARW, No-DA)": ModelConfiguration.short_range_puertorico_no_da
    }
}
"""
Mapping from ModelDomain to pretty string representations of model configurations.
Pretty strings map to model ModelConfiguration enums.
"""

METRIC_STRINGS: dict[str, Metric] = {
    "Nash-Sutcliffe Model Efficiency": Metric.nash_sutcliffe_efficiency,
    "Relative mean bias": Metric.relative_mean_bias,
    "Pearson correlation coefficient": Metric.pearson_correlation_coefficient,
    "Relative mean": Metric.relative_mean,
    "Relative standard deviation": Metric.relative_standard_deviation,
    "Kling-Gupta Model Efficiency": Metric.kling_gupta_efficiency
}
"""Mapping from pretty strings to column names."""

CONFIDENCE_STRINGS: dict[str, str] = {
    "Point": "_point",
    "Lower": "_lower",
    "Upper": "_upper"
}
"""Mapping from pretty strings to column suffixes."""

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

@dataclass
class FilterState:
    """Filter state variables."""
    evaluation: str
    domain: ModelDomain
    configuration: ModelConfiguration
    threshold: str
    metric: Metric
    metric_label: str
    confidence: str
    lead_time: int

    def __eq__(self, other):
        self_states = (
            self.evaluation,
            self.domain,
            self.configuration,
            self.threshold,
            self.metric,
            self.metric_label,
            self.confidence,
            self.lead_time
        )
        other_states = (
            other.evaluation,
            other.domain,
            other.configuration,
            other.threshold,
            other.metric,
            other.metric_label,
            other.confidence,
            other.lead_time
        )
        for s, o in zip(self_states, other_states):
            if s != o:
                return False
        return True

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
    double_click = "double_click"

class FilteringWidgets:
    def __init__(self, evaluation_registry: EvaluationRegistry):
        # Filtering options
        self.callbacks: list[Callable] = []
        self.evaluation_filter = pn.widgets.Select(
            name="Evaluation",
            options=list(evaluation_registry.evaluations.keys())
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
        evaluation = self.evaluation_filter.value
        domain = DOMAIN_STRINGS[self.domain_filter.value]
        return FilterState(
            evaluation=evaluation,
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
