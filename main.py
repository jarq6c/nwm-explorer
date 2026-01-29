"""Test script."""
from pathlib import Path
from typing import Literal

from nwm_explorer.constants import ModelConfiguration, EvaluationMetric, GROUP_SPECIFICATIONS
from nwm_explorer.plotter import plot_preprocess, plot_maps, NoDataError
from nwm_explorer.configuration import load_configuration

def plot(
        root: Path = Path("/ised/nwm_explorer_data"),
        label: str = "FY2024-FY2025",
        threshold: str = "q85_cfs",
        rank: Literal["min", "median", "max"] = "max",
        model_title: str = "National Water Model v3.0",
        model_domain: str = "CONUS",
        app_config: Path = Path("config.json")
    ) -> None:
    """
    Plot evaluation maps.
    """
    # Load app configuration
    app_config = load_configuration(configuration_file=app_config)

    # Process each model configuration
    for c in list(ModelConfiguration):
        # Get lead times
        specs = GROUP_SPECIFICATIONS[c]
        lead_times = range(
            0,
            specs.lead_time_hours_max+specs.window_interval,
            specs.window_interval
            )

        # Process each lead time
        for l in lead_times:
            # Process each metric
            for m in list(EvaluationMetric):
                # Get parameters
                try:
                    parameters = plot_preprocess(
                        root=root,
                        label=label,
                        configuration=c,
                        metric=m,
                        threshold=threshold,
                        lead_time_hours_min=l,
                        api_key=app_config.stadia_api_key,
                        rank=rank,
                        model_title=model_title,
                        model_domain=model_domain
                    )
                except NoDataError as e:
                    print(e)
                    continue

                # Plot metrics by region
                plot_maps(parameters)

if __name__ == "__main__":
    plot()
