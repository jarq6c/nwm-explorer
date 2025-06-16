"""Read-only methods."""
from pathlib import Path
from dataclasses import dataclass
from typing import TypedDict
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from nwm_explorer.mappings import Domain, Configuration, Metric, Confidence

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
