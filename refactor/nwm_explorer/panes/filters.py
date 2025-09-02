"""
Filters used by evaluation dashboard to specify data sources.
"""
from pathlib import Path

import polars as pl
import panel as pn
from panel.viewable import Viewer

from nwm_explorer.application.api import (
    EvaluationRegistry, ModelDomain, ModelConfiguration)

class Filters(Viewer):
    """
    Houses main filtering widgets and tracks their state.

    Attributes
    ----------
    evaluation: str
        Currently selected evaluation.
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
        self.configuration_selector = pn.widgets.Select(
            name="Model configuration",
            options=self.configuration_options
        )

        # Callbacks
        def update_selector_options(evaluation: str) -> None:
            if evaluation is None:
                return
            self.domain_selector.options = self.domain_options
            self.configuration_selector.options = self.configuration_options
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
    def configuration_options(self) -> list[ModelConfiguration]:
        """List of configuration keys."""
        return [ModelConfiguration(c) for c in self.registry.evaluations[self.evaluation][self.domain]]

    @property
    def configuration(self) -> ModelConfiguration:
        """Currently selected configuration."""
        return ModelConfiguration(self.configuration_selector.value)
    
    @property
    def filepath(self) -> Path:
        return Path(self.registry.evaluations[self.evaluation][self.domain][self.configuration])

    @property
    def dataframe(self) -> pl.LazyFrame:
        return pl.scan_parquet(self.filepath)

    def __panel__(self) -> pn.Card:
        return pn.Card(
            pn.Column(
                self.evaluation_selector,
                self.domain_selector,
                self.configuration_selector
            ),
            title="Filters",
            collapsible=False
        )
