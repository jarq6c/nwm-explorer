"""Test script."""
from pathlib import Path
from typing import Literal

from nwm_explorer.constants import ModelConfiguration, EvaluationMetric
from nwm_explorer.plotter import plot_preprocess, plot_map
from nwm_explorer.configuration import load_configuration

def plot(
        root: Path = Path("/ised/nwm_explorer_data"),
        label: str = "FY2024Q1",
        configuration: ModelConfiguration = "medium_range_mem1",
        metric: EvaluationMetric = "relative_mean",
        # metric: EvaluationMetric = "nash_sutcliffe_efficiency",
        # threshold: str = "None",
        threshold: str = "q85_cfs",
        lead_time_hours_min: int = 72,
        rank: Literal["min", "median", "max"] = "max",
        title: str = "High Flow Evaluation",
        model_title: str = "National Water Model v3.0",
        model_domain: str = "CONUS",
        app_config: Path = Path("config.json")
    ) -> None:
    """
    Plot evaluation results on a map.
    """
    # Load app configuration
    app_config = load_configuration(configuration_file=app_config)

    # Load data
    parameters = plot_preprocess(
        root=root,
        label=label,
        configuration=configuration,
        metric=metric,
        threshold=threshold,
        lead_time_hours_min=lead_time_hours_min,
        api_key=app_config.stadia_api_key,
        rank=rank,
        title=title,
        model_title=model_title,
        model_domain=model_domain
    )

    # Plot metrics
    plot_map(parameters)

if __name__ == "__main__":
    plot()
