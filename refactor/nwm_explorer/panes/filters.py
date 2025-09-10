"""
Filters used by evaluation dashboard to specify data sources.
"""
from pathlib import Path

import polars as pl
import panel as pn
from panel.viewable import Viewer

from nwm_explorer.application.api import (
    EvaluationRegistry, ModelDomain, ModelConfiguration, ModelForcing,
    DOMAIN_FORCING_CONFIGURATION, Threshold, Metric, Confidence)

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
            name="Model domain",
            options=self.domain_options
        )
        self.forcing_selector = pn.widgets.Select(
            name="Model forcing",
            options=self.forcing_options
        )
        self.threshold_selector = pn.widgets.Select(
            name="Streamflow threshold (≥)",
            options=list(Threshold)
        )
        self.metric_selector = pn.widgets.Select(
            name="Evaluation metric",
            options=list(Metric)
        )
        self.confidence_selector = pn.widgets.Select(
            name="Confidence estimate (95%)",
            options=list(Confidence)
        )

        # Callbacks
        def update_selector_options(evaluation: str) -> None:
            if evaluation is None:
                return
            self.domain_selector.options = self.domain_options
            self.forcing_selector.options = self.forcing_options
        pn.bind(update_selector_options, self.evaluation_selector.param.value, watch=True)

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
    def filepath(self) -> Path:
        """Currently selected file path."""
        return Path(self.registry.evaluations[self.evaluation][self.domain][self.forcing])
    
    @property
    def configuration(self) -> ModelConfiguration:
        """Currently selected model configuration."""
        return DOMAIN_FORCING_CONFIGURATION[self.domain][self.forcing]

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
                self.confidence_selector
            ),
            title="Filters",
            collapsible=False
        )
