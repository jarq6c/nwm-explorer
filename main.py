"""Test script."""
from pathlib import Path
from typing import Literal

from nwm_explorer.constants import ModelConfiguration, EvaluationMetric
from nwm_explorer.plotter import plotting_preprocess, map_metrics

def plot(
        root: Path = Path("/ised/nwm_explorer_data"),
        label: str = "FY2024Q1",
        configuration: ModelConfiguration = "short_range",
        metric: EvaluationMetric = "nash_sutcliffe_efficiency",
        threshold: str = "q85_cfs",
        lead_time_hours_min: int = 6,
        rank: Literal["min", "median", "max"] = "max"
    ) -> None:
    """
    Plot evaluation results on a map.
    """
    data = plotting_preprocess(
        root=root,
        label=label,
        configuration=configuration,
        metric=metric,
        threshold=threshold,
        lead_time_hours_min=lead_time_hours_min,
        rank=rank
    )

    # Plot metrics
    map_metrics(data=data)

if __name__ == "__main__":
    plot()
