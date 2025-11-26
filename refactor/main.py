"""Command-line Interface."""
from pathlib import Path
from datetime import datetime

import typer
import pandas as pd

from modules.configuration import load_configuration
from modules.nwm import download_nwm
from modules.usgs import download_usgs
from modules.pairs import pair_nwm_usgs
from modules.evaluate import evaluate as run_evaluation
from modules.gui import serve_dashboards

app = typer.Typer()
"""Main typer command-line application."""

@app.command()
def download(
    start: datetime,
    end: datetime,
    root: Path,
    jobs: int = 1,
    retries: int = 3
    ) -> None:
    """
    Download and process NWM and USGS data for evaluations.
    """
    # Use time stamps
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    # NWM downloads
    download_nwm(
        start=start,
        end=end,
        root=root,
        jobs=jobs,
        retries=retries
    )

    # USGS downloads
    download_usgs(
        start=start-pd.Timedelta("1d"),
        end=end+pd.Timedelta("10d"),
        root=root,
        retries=retries
    )

@app.command()
def pair(
    start: datetime,
    end: datetime,
    root: Path
    ) -> None:
    """
    Resample and pair NWM predictions to USGS observations.
    """
    # Use timestamps
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    # Pair
    pair_nwm_usgs(
        root=root,
        start_date=start,
        end_date=end
    )

@app.command()
def compute(
    start: datetime,
    end: datetime,
    root: Path,
    label: str,
    jobs: int = 1,
    sites_per_chunk: int = 500
    ) -> None:
    """
    Compute evaluation metrics for NWM-USGS pairs.
    """
    # Use timestamps
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    # Evaluate
    run_evaluation(
        label=label,
        root=root,
        start_time=start,
        end_time=end,
        processes=jobs,
        sites_per_chunk=sites_per_chunk
    )

@app.command()
def evaluate(configuration: Path = Path("config.json")) -> None:
    """
    Run standard evaluation including download, pair, and compute. Parameters
    set in configuration file.
    """
    # Load configuration
    config = load_configuration(configuration_file=configuration)

    # Process each evaluation
    for e in config.evaluations:
        # Set start and end times
        start = pd.Timestamp(e.start_time)
        end = pd.Timestamp(e.end_time)

        # NWM downloads
        download_nwm(
            start=start,
            end=end,
            root=config.root,
            jobs=config.processes,
            retries=config.retries
        )

        # USGS downloads
        download_usgs(
            start=start-pd.Timedelta("1d"),
            end=end+pd.Timedelta("10d"),
            root=config.root,
            retries=config.retries
        )

        # Pair
        pair_nwm_usgs(
            root=config.root,
            start_date=start,
            end_date=end
        )

        # Evaluate
        run_evaluation(
            label=e.label,
            root=config.root,
            start_time=start,
            end_time=end,
            processes=config.processes,
            sites_per_chunk=config.sites_per_chunk
        )

@app.command()
def display(configuration: Path = Path("config.json")) -> None:
    """
    Launch graphical application in the browser. Shutdown the application using
    ctrl+c.
    """
    serve_dashboards(
        configuration_file=configuration
    )

# import pandas as pd

# from modules.evaluate import evaluate
# from modules.usgs import download_usgs
# from modules.nwm import download_nwm
# from modules.pairs import pair_nwm_usgs

# def main() -> None:
#     """Main."""
#     serve_dashboards(
#         configuration_file=Path("config.json")
#     )
#     # root = Path("/ised/nwm_explorer_data")

#     # download_usgs(
#     #     start=pd.Timestamp("2023-09-28"),
#     #     end=pd.Timestamp("2025-10-11"),
#     #     root=root
#     # )

#     # download_nwm(
#     #     start=pd.Timestamp("2023-10-01"),
#     #     end=pd.Timestamp("2025-09-30"),
#     #     root=root,
#     #     jobs=18,
#     #     retries=1
#     # )

#     # pair_nwm_usgs(
#     #     root=root,
#     #     start_date=pd.Timestamp("2023-10-01"),
#     #     end_date=pd.Timestamp("2025-09-30")
#     # )

#     # evaluations = {
#     #     "FY2024Q1": (pd.Timestamp("2023-10-01"), pd.Timestamp("2023-12-31T23:59")),
#     #     "FY2024Q2": (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-03-31T23:59")),
#     #     "FY2024Q3": (pd.Timestamp("2024-04-01"), pd.Timestamp("2024-06-30T23:59")),
#     #     "FY2024Q4": (pd.Timestamp("2024-07-01"), pd.Timestamp("2024-09-30T23:59")),
#     #     "FY2025Q1": (pd.Timestamp("2024-10-01"), pd.Timestamp("2024-12-31T23:59")),
#     #     "FY2025Q2": (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-03-31T23:59")),
#     #     "FY2025Q3": (pd.Timestamp("2025-04-01"), pd.Timestamp("2025-06-30T23:59")),
#     #     "FY2025Q4": (pd.Timestamp("2025-07-01"), pd.Timestamp("2025-09-30T23:59")),
#     #     "FY2024": (pd.Timestamp("2023-10-01"), pd.Timestamp("2024-09-30T23:59")),
#     #     "FY2025": (pd.Timestamp("2024-10-01"), pd.Timestamp("2025-09-30T23:59")),
#     #     "FY2024-FY2025": (pd.Timestamp("2023-10-01"), pd.Timestamp("2025-09-30T23:59"))
#     # }

#     # for l, (s, e) in evaluations.items():
#     #     evaluate(
#     #         label = l,
#     #         root = root,
#     #         start_time = s,
#     #         end_time = e,
#     #         processes = 18,
#     #         sites_per_chunk = 500
#     #     )
#     # evals = load_metrics(
#     #     root=root,
#     #     label="FY2024Q3",
#     #     configuration=ModelConfiguration.MEDIUM_RANGE_MEM_1,
#     #     metric=Metric.KLING_GUPTA_EFFICIENCY
#     # )
#     # print(evals)

if __name__ == "__main__":
    # Run the CLI.
    app()
