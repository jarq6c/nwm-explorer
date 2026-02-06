"""Test script."""
from pathlib import Path
from typing import Literal

from nwm_explorer.constants import ModelConfiguration, EvaluationMetric
from nwm_explorer.plotter import plot_preprocess, plot_maps, NoDataError
from nwm_explorer.configuration import load_configuration

def plot(
        root: Path = Path("/ised/nwm_explorer_data"),
        label: str = "FY2024-FY2025",
        threshold: str = "q85_cfs",
        rank: Literal["min", "median", "max"] = "max",
        configuration: ModelConfiguration = ModelConfiguration.SHORT_RANGE,
        metric: EvaluationMetric = EvaluationMetric.NASH_SUTCLIFFE_EFFICIENCY,
        minimum_lead_time_hours: int = 0,
        model_title: str = "National Water Model v3.0",
        app_config: Path = Path("config.json")
    ) -> None:
    """
    Plot an evaluation map.
    """
    # Load app configuration
    app_config = load_configuration(configuration_file=app_config)

    try:
        # Get parameters
        parameters = plot_preprocess(
            root=root,
            label=label,
            configuration=configuration,
            metric=metric,
            threshold=threshold,
            lead_time_hours_min=minimum_lead_time_hours,
            api_key=app_config.stadia_api_key,
            rank=rank,
            model_title=model_title
        )

        # Plot metrics by region
        plot_maps(parameters)
    except NoDataError as e:
        print(e)

if __name__ == "__main__":
    plot()
