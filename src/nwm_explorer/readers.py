"""Read-only methods."""
from pathlib import Path
from dataclasses import dataclass
import pandas as pd

from nwm_explorer.mappings import Domain, Configuration, Metric, Confidence

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
