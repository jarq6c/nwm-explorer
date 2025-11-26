"""Command-line Interface."""
from pathlib import Path
from datetime import datetime
from typing import Literal
from sys import stdout

import typer
import pandas as pd
import polars as pl

from modules.configuration import load_configuration
from modules.nwm import download_nwm
from modules.usgs import download_usgs
from modules.pairs import pair_nwm_usgs
from modules.evaluate import evaluate as run_evaluation
from modules.evaluate import scan_evaluations
from modules.constants import ModelConfiguration, Metric, COLUMN_DESCRIPTIONS
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
def export(
        root: Path,
        label: str,
        configuration: ModelConfiguration,
        metric: Metric,
        output: typer.FileTextWrite | None = None,
        lead_time_hours_min: int = 0,
        rank: Literal["min", "median", "max"] = "median",
        additional_columns: tuple[str] | None = None,
        no_data_value: str | None = None
    ) -> None:
    """
    Export evaluation metrics to CSV.
    """
    # Set output
    if output is None:
        output = stdout

    # Set additional columns
    if additional_columns is None:
        additional_columns = [
            "label",
            "configuration",
            "nwm_feature_id",
            "usgs_site_code",
            "reference_time_min",
            "reference_time_max",
            "lead_time_hours_min",
            "predicted_value_time_min",
            "predicted_value_time_max",
            f"predicted_cfs_{rank}",
            "observed_value_time_min",
            "observed_value_time_max",
            f"observed_cfs_{rank}",
            "sample_size"
        ]

    # Load metrics
    data = scan_evaluations(
        root
    ).filter(
        pl.col("label") == label,
        pl.col("configuration") == configuration,
        pl.col("lead_time_hours_min") == lead_time_hours_min
    ).select(
        additional_columns + [
            f"{metric}_{rank}_lower",
            f"{metric}_{rank}_point",
            f"{metric}_{rank}_upper"
        ]
    ).collect()

    # Write header
    header = "# National Water Model Evaluations\n# \n"
    for col in data.columns:
        header += f"# {col}: {COLUMN_DESCRIPTIONS.get(col)}\n"
    header += "# \n"
    output.write(header)

    # Write data
    output.write(data.write_csv(
        float_precision=2,
        datetime_format="%Y-%m-%dT%H:%M",
        null_value=no_data_value
    ))

@app.command()
def display(configuration: Path = Path("config.json")) -> None:
    """
    Launch graphical application in the browser. Shutdown the application using
    ctrl+c.
    """
    serve_dashboards(
        configuration_file=configuration
    )

if __name__ == "__main__":
    # Run the CLI.
    app()
