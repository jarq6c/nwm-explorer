"""Filtering widgets."""
from dataclasses import dataclass
from typing import Callable
import pandas as pd
import panel as pn
from nwm_explorer.mappings_new import (
    EVALUATION_STRINGS,
    DOMAIN_STRINGS,
    DOMAIN_CONFIGURATIONS,
    METRIC_STRINGS,
    CONFIDENCE_STRINGS,
    LEAD_TIME_VALUES,
    EVALUATION_PERIODS,
    Evaluation,
    Domain,
    Configuration,
    Metric,
    Confidence,
    CallbackType
)

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
