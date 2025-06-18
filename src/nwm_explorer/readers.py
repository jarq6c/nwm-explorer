"""Read-only methods."""
from pathlib import Path
from dataclasses import dataclass
from typing import TypedDict
import numpy as np
import pandas as pd
import polars as pl
import plotly.graph_objects as go

from nwm_explorer.mappings import (Domain, Configuration, Metric, Confidence,
    NWM_URL_BUILDERS, FileType, Variable, Units)
from nwm_explorer.urls import generate_reference_dates
from nwm_explorer.data import generate_filepath
from nwm_explorer.logger import get_logger

def read_NWM_output(
        root: Path,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        low_memory: bool = False
) -> dict[tuple[Domain, Configuration], pl.LazyFrame]:
    """
    Find and lazily load NWM output.

    Parameters
    ----------
    root: Path, required
        Root directory to save downloaded and processed files.
    start_date: pd.Timestamp
        First date to start retrieving data.
    end_date: pd.Timestamp
        Last date to retrieve data.
    low_memory: bool, default False
        Reduce memory pressure at the expense of performance.
    
    Returns
    -------
    dict[tuple[Domain, Configuration], pl.LazyFrame]
    """
    logger = get_logger("nwm_explorer.readers.read_NWM_output")
    reference_dates = generate_reference_dates(start_date, end_date)

    # Download and process model output
    model_output = {}
    for (domain, configuration), _ in NWM_URL_BUILDERS.items():
        day_files = []
        for rd in reference_dates:
            # Check for file existence
            parquet_file = generate_filepath(
                root, FileType.parquet, configuration, Variable.streamflow,
                Units.cubic_feet_per_second, rd
            )
            logger.info(f"Building {parquet_file}")
            if parquet_file.exists():
                logger.info(f"Found existing {parquet_file}")
                day_files.append(parquet_file)
                continue
            
        # Check for at least one file
        if not day_files:
            logger.info(f"Found no data for {domain} {configuration}")
            continue
        
        # Merge files
        logger.info(f"Merging parquet files")
        model_output[(domain, configuration)] = pl.scan_parquet(day_files,
            low_memory=low_memory)
    return model_output

def read_USGS_observations(
        root: Path,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        routelinks: dict[Domain, pl.LazyFrame]
) -> dict[Domain, pl.LazyFrame]:
    """
    Download and process USGS observations.

    Parameters
    ----------
    root: Path, required
        Root directory to save downloaded and processed files.
    start_date: pd.Timestamp
        First date to start retrieving data.
    end_date: pd.Timestamp
        Last date to retrieve data.
    routelinks: dict[Domain, LazyFrame]
        Mapping from Domain to crosswalk data.
    
    Returns
    -------
    dict[Domain, pl.LazyFrame]
    """
    logger = get_logger("nwm_explorer.pipelines.load_USGS_observations")
    # Download and process model output
    observations = {}
    s = start_date.strftime("%Y%m%dT%H")
    e = end_date.strftime("%Y%m%dT%H")
    for domain, _ in routelinks.items():
        # Check for file existence
        parquet_directory = root / FileType.parquet / Configuration.usgs
        parquet_file = parquet_directory / f"{domain}_{s}_{e}.parquet"
        logger.info(f"Building {parquet_file}")
        if parquet_file.exists():
            logger.info(f"Found existing {parquet_file}")
            observations[domain] = pl.scan_parquet(parquet_file)
    return observations

def read_pairs(
        root: Path,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp
) -> dict[tuple[Domain, Configuration], pl.LazyFrame]:
    """
    Find and lazily load paired data.

    Parameters
    ----------
    root: Path, required
        Root directory to save downloaded and processed files.
    start_date: pd.Timestamp
        First date to start retrieving data.
    end_date: pd.Timestamp
        Last date to retrieve data.
    
    Returns
    -------
    dict[tuple[Domain, Configuration], pl.LazyFrame]
    """
    logger = get_logger("nwm_explorer.readers.read_pairs")
    # Process model output
    logger.info("Scanning pairs")
    reference_dates = generate_reference_dates(start_date, end_date)
    pairs = {}
    for (domain, configuration), _ in NWM_URL_BUILDERS.items():
        day_files = []
        for rd in reference_dates:
            # Check for file existence
            parquet_file = generate_filepath(
                root, FileType.parquet, configuration, Variable.streamflow_pairs,
                Units.cubic_feet_per_second, rd
            )
            if parquet_file.exists():
                logger.info(f"Found existing file: {parquet_file}")
                day_files.append(parquet_file)
                continue
            
        # Check for at least one file
        if not day_files:
            logger.info(f"Found no data for {domain} {configuration}")
            continue
        
        # Merge files
        logger.info(f"Merging parquet files")
        pairs[(domain, configuration)] = pl.scan_parquet(day_files)
    return pairs

class FigurePatch(TypedDict):
    """
    A plotly figure patch.

    Keys
    ----
    data: list[go.Trace]
        A list of plotly traces.
    layout: go.Layout
        Plotly layout.
    """
    data: list[go.Trace]
    layout: go.Layout

@dataclass
class DashboardState:
    """Dashboard state variables."""
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    domain: Domain
    configuration: Configuration
    threshold: str
    metric: Metric
    confidence: Confidence
    lead_time: int

@dataclass
class MetricReader:
    """Intermediate metric reader to query and return data to dashboards."""
    root: Path

    def query(self, state: DashboardState) -> int:
        """Return data matching dashboard state."""
        return str(state).replace(",", "<br>")
    
    def get_plotly_patch(self, state: DashboardState) -> FigurePatch:
        """Return map of sites matching dashboard state."""
        xx = np.linspace(-3.5, 3.5, 100)
        yy = np.linspace(-3.5, 3.5, 100)
        x, y = np.meshgrid(xx, yy)
        z = np.exp(-((x - 1) ** 2) - y**2) - (x**3 + y**4 - x / 5) * np.exp(-(x**2 + y**2))

        surface = go.Surface(z=z)
        layout = go.Layout(
            title=str(state.domain),
            autosize=False,
            width=500,
            height=500,
            margin=dict(t=50, b=50, r=50, l=50)
        )

        return dict(data=[surface], layout=layout)
