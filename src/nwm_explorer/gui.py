"""Generate and serve exploratory applications."""
from pathlib import Path
from dataclasses import dataclass
import panel as pn
from panel.template import BootstrapTemplate

@dataclass
class DashboardState:
    """Dashboard state variables."""
    evaluation: str
    domain: str
    configuration: str
    threshold: str
    metric: str
    confidence: str
    lead_time: int

class Dashboard:
    """Build a dashboard for exploring National Water Model output."""
    def __init__(self, root: Path, title: str):
        # Filtering options
        self.evaluation_filter = pn.widgets.Select(
            name="Evaluation",
            options=[
                "FY2024_Q1",
                "FY2024_Q2"
            ]
        )
        self.domain_filter = pn.widgets.Select(
            name="Model Domain",
            options=[
                "Alaska",
                "CONUS",
                "Hawaii",
                "Puerto Rico"
            ]
        )
        self.configuration_filter = pn.widgets.Select(
            name="Model Configuration",
            options=["Analysis", "Medium Range"]
            )
        self.threshold_filter = pn.widgets.Select(
            name="Streamflow Threshold (≥)",
            options=[
                "0th Percentile (All data)",
                "85th Percentile",
                "95th Percentile",
                "99th Percentile"
            ]
        )
        self.metric_filter = pn.widgets.Select(
            name="Evaluation Metric",
            options=[
                "Mean relative bias",
                "Nash-Sutcliffe efficiency",
                "Pearson correlation coefficient",
                "Kling-Gupta efficiency"
            ]
        )
        self.confidence_filter = pn.widgets.Select(
            name="Confidence Estimate (95%)",
            options=[
                "Point",
                "Lower",
                "Upper"
            ]
        )
        self.lead_time_filter = pn.widgets.DiscretePlayer(
            name="Minimum lead time (hours)",
            options=[2, 3, 4, 5],
            show_loop_controls=False,
            visible_buttons=["previous", "next"],
            width=300
            )

        # Layout filtering options
        self.filter_card = pn.Card(
            pn.Column(
                self.evaluation_filter,
                self.domain_filter,
                self.configuration_filter,
                self.threshold_filter,
                self.metric_filter,
                self.confidence_filter,
                self.lead_time_filter
                ),
            title="Filters",
            collapsible=False
            )

        # Layout cards
        layout = pn.Row(self.filter_card)
        self.template = BootstrapTemplate(title=title)
        self.template.main.append(layout)

def generate_dashboard(
        root: Path,
        title: str
        ) -> BootstrapTemplate:
    return Dashboard(root, title).template

def generate_dashboard_closure(
        root: Path,
        title: str
        ) -> BootstrapTemplate:
    def closure():
        return generate_dashboard(root, title)
    return closure

def serve_dashboard(
        root: Path,
        title: str
        ) -> None:
    # Slugify title
    slug = title.lower().replace(" ", "-")

    # Serve
    endpoints = {
        slug: generate_dashboard_closure(root, title)
    }
    pn.serve(endpoints)
