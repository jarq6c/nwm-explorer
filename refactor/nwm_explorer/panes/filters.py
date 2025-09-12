"""
Filters used by evaluation dashboard to specify data sources.
"""
from pathlib import Path
from typing import Any, Callable

import polars as pl
import panel as pn
from panel.viewable import Viewer

from nwm_explorer.application.api import (
    EvaluationRegistry, ModelDomain, ModelConfiguration, ModelForcing,
    DOMAIN_FORCING_CONFIGURATION, Threshold, Metric, Confidence,
    LEAD_TIME_VALUES)

class EditablePlayer(Viewer):
    """
    DiscretePlayer that refreshes when changing options.

    Parameters
    ----------
    params: any
        Keyword arguments passed to pn.widgets.DiscretePlayer.
    """
    def __init__(
            self,
            **params
        ) -> None:
        # Initialize
        self._params: dict[str, Any] = params
        self._container = pn.pane.Placeholder(
            pn.widgets.DiscretePlayer(**params)
        )
        self._watchers: list[Callable[[Any], None]] = []

    def __panel__(self) -> pn.pane.Placeholder:
        return self._container
    
    @property
    def value(self) -> Any:
        return self._container.object.value
    
    def add_watcher(self, func: Callable[[Any], None]) -> None:
        self._watchers.append(func)
        pn.bind(func, self._container.object.param.value, watch=True)

    def update(self, **params) -> None:
        """
        Use this method to update underlying parameters of DiscretePlayer.
        """
        # Handle options
        if ("options" in params) and ("value" not in params):
            values = params["options"]
            v = self._container.object.value
            self._params.update({"value": v if v in values else values[0]})

        # Apply remaining updates
        self._params.update(params)

        # Refresh widget
        self._container.object = pn.widgets.DiscretePlayer(**self._params)

        # Transfer callbacks
        for w in self._watchers:
            pn.bind(w, self._container.object.param.value, watch=True)

class Filters(Viewer):
    """
    Houses main filtering widgets and tracks their state.

    Parameters
    ----------
    registry: EvaluationRegistry
        EvaluationRegistry used throughout dashboard.
    params: any
        Additional keyword arguments passed directly to panel.viewable.Viewer.
    """
    def __init__(self, registry: EvaluationRegistry, **params) -> None:
        super().__init__(**params)

        # Reference registry
        self.registry = registry

        # Build widgets
        self.evaluation_selector = pn.widgets.Select(
            name="Evaluation",
            options=self.evaluation_options
        )
        self.domain_selector = pn.widgets.Select(
            name="Model Domain",
            options=self.domain_options
        )
        self.forcing_selector = pn.widgets.Select(
            name="Model Forcing",
            options=self.forcing_options
        )
        self.threshold_selector = pn.widgets.Select(
            name="Streamflow Threshold (≥)",
            options=list(Threshold)
        )
        self.metric_selector = pn.widgets.Select(
            name="Evaluation Metric",
            options=list(Metric)
        )
        self.confidence_selector = pn.widgets.Select(
            name="Confidence Estimate (95%)",
            options=list(Confidence)
        )

        # Initialize lead time state
        lead_time_options = LEAD_TIME_VALUES.get(self.configuration, [0])
        self.lead_time_selector = EditablePlayer(
            name="Minimum lead time (hours)",
            options=lead_time_options,
            show_loop_controls=False,
            visible_buttons=["previous", "next"],
            width=300,
            value=lead_time_options[0]
        )

        # Domain callback
        def update_domain_options(evaluation: str) -> None:
            if evaluation is None:
                return
            self.domain_selector.options = self.domain_options
        pn.bind(update_domain_options, self.evaluation_selector.param.value, watch=True)

        # Forcing callback
        def update_forcing_options(event: str) -> None:
            if event is None:
                return
            self.forcing_selector.options = self.forcing_options
        pn.bind(update_forcing_options, self.domain_selector.param.value, watch=True)
        pn.bind(update_forcing_options, self.evaluation_selector.param.value, watch=True)

        # Lead time callback
        def update_lead_times(event: str) -> None:
            if event is None:
                return
            lead_time_options = LEAD_TIME_VALUES.get(self.configuration, [0])
            self.lead_time_selector.update(options=lead_time_options)
        pn.bind(update_lead_times, self.forcing_selector.param.value, watch=True)
        pn.bind(update_lead_times, self.domain_selector.param.value, watch=True)
        pn.bind(update_lead_times, self.evaluation_selector.param.value, watch=True)

    @property
    def evaluation_options(self) -> list[str]:
        """List of evaluation keys."""
        return list(self.registry.evaluations.keys())

    @property
    def evaluation(self) -> str:
        """Currently selected evaluation."""
        return self.evaluation_selector.value

    @property
    def domain_options(self) -> list[ModelDomain]:
        """List of domain keys."""
        return [ModelDomain(d) for d in self.registry.evaluations[self.evaluation]]

    @property
    def domain(self) -> ModelDomain:
        """Currently selected domain."""
        return ModelDomain(self.domain_selector.value)

    @property
    def forcing_options(self) -> list[ModelForcing]:
        """List of forcing keys."""
        return [ModelForcing(c) for c in self.registry.evaluations[self.evaluation][self.domain]]

    @property
    def forcing(self) -> ModelForcing:
        """Currently selected forcing."""
        return ModelForcing(self.forcing_selector.value)

    @property
    def threshold(self) -> Threshold:
        """Currently selected threshold."""
        return Threshold(self.threshold_selector.value)

    @property
    def metric(self) -> Metric:
        """Currently selected metric."""
        return Metric(self.metric_selector.value)

    @property
    def confidence(self) -> Confidence:
        """Currently selected confidence boundary."""
        return Confidence(self.confidence_selector.value)

    @property
    def filepath(self) -> Path:
        """Currently selected file path."""
        return Path(self.registry.evaluations[self.evaluation][self.domain][self.forcing])
    
    @property
    def configuration(self) -> ModelConfiguration:
        """Currently selected model configuration."""
        return DOMAIN_FORCING_CONFIGURATION[self.domain][self.forcing]

    @property
    def lead_time(self) -> int:
        """Currently selected lead time value."""
        return self.lead_time_selector.value

    @property
    def dataframe(self) -> pl.LazyFrame:
        """Scans current file path and returns lazy dataframe."""
        return pl.scan_parquet(self.filepath)

    def __panel__(self) -> pn.Card:
        return pn.Card(
            pn.Column(
                self.evaluation_selector,
                self.domain_selector,
                self.forcing_selector,
                self.threshold_selector,
                self.metric_selector,
                self.confidence_selector,
                self.lead_time_selector
            ),
            title="Filters",
            collapsible=False
        )
