"""Generate and serve exploratory applications."""
from pathlib import Path
from dataclasses import dataclass
import panel as pn
from panel.template import BootstrapTemplate

from nwm_explorer.mappings import (DOMAIN_MAPPING, DOMAIN_CONFIGURATION_MAPPING,
    LEAD_TIME_VALUES, Domain, Configuration, Confidence)

@dataclass
class DashboardState:
    """Dashboard state variables."""
    evaluation: str
    domain: Domain
    configuration: Configuration
    threshold: str
    metric: str
    confidence: Confidence
    lead_time: int

class Dashboard:
    """Build a dashboard for exploring National Water Model output."""
    def __init__(self, root: Path, title: str):
        # Domain and configuration filtering options
        self.domain_filter = pn.widgets.Select(
            name="Model Domain",
            options=[
                "Alaska",
                "CONUS",
                "Hawaii",
                "Puerto Rico"
            ],
            value="Alaska"
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
        self.evaluation_filter = pn.widgets.Select(
            name="Evaluation",
            options=[
                "FY2024_Q1",
                "FY2024_Q2"
            ]
        )
        current_domain = DOMAIN_MAPPING[self.domain_filter.value]
        configurations = list(
            DOMAIN_CONFIGURATION_MAPPING[current_domain].keys())
        self.configuration_filter = pn.widgets.Select(
            name="Model Configuration",
            options=configurations,
            value=configurations[0]
            )
        self.current_configuration = (
            DOMAIN_CONFIGURATION_MAPPING[current_domain][self.configuration_filter.value])
        lead_times = LEAD_TIME_VALUES.get(self.current_configuration, [0])
        lead_time_filter = pn.widgets.DiscretePlayer(
            name="Minimum lead time (hours)",
            options=lead_times,
            value=lead_times[0],
            show_loop_controls=False,
            visible_buttons=["previous", "next"],
            width=300
            )
        self.current_lead_time = lead_times[0]
        
        # Domain callbacks
        def domain_update(domain: str) -> None:
            if domain is None:
                return
            self.handle_domain_update(domain)
        pn.bind(domain_update, self.domain_filter, watch=True)

        # Configuration callbacks
        def configuration_update(configuration: str) -> None:
            if configuration is None:
                return
            self.handle_configuration_update(configuration)
        pn.bind(configuration_update, self.configuration_filter, watch=True)

        # Lead time callbacks
        def lead_time_update(lead_time: int) -> None:
            if lead_time is None:
                return
            self.handle_lead_time_update(lead_time)
        pn.bind(lead_time_update, lead_time_filter, watch=True)

        # Layout filtering options
        self.filter_card = pn.Card(
            pn.Column(
                self.evaluation_filter,
                self.domain_filter,
                self.configuration_filter,
                self.threshold_filter,
                self.metric_filter,
                self.confidence_filter,
                lead_time_filter
                ),
            title="Filters",
            collapsible=False
            )

        # Layout cards
        grid = pn.Row(self.filter_card)
        self.template = BootstrapTemplate(title=title)
        self.template.main.append(grid)
    
    def handle_domain_update(self, domain: str) -> None:
        # Update configuration options
        current_domain = DOMAIN_MAPPING[domain]
        configurations = list(
            DOMAIN_CONFIGURATION_MAPPING[current_domain].keys()
            )
        self.configuration_filter.options = []
        self.configuration_filter.options = configurations
    
    def handle_configuration_update(self, configuration: str) -> None:
        # Update current configuration
        self.current_configuration = (
            DOMAIN_CONFIGURATION_MAPPING[self.current_domain][configuration]
            )
        print(self.current_configuration)

        # Update lead time options
        lead_times = LEAD_TIME_VALUES.get(self.current_configuration, [0])
        if self.current_lead_time in lead_times:
            current_lead_time = self.current_lead_time
        else:
            current_lead_time = lead_times[0]
        lead_time_filter = pn.widgets.DiscretePlayer(
            name="Minimum lead time (hours)",
            options=lead_times,
            value=current_lead_time,
            show_loop_controls=False,
            visible_buttons=["previous", "next"],
            width=300
            )
        def lead_time_update(lead_time: int) -> None:
            if lead_time is None:
                return
            self.handle_lead_time_update(lead_time)
        pn.bind(lead_time_update, lead_time_filter, watch=True)
        self.filter_card[0][-1] = lead_time_filter
        self.handle_lead_time_update(current_lead_time)
    
    def handle_lead_time_update(self, lead_time: int) -> None:
        self.current_lead_time = lead_time
        print(self.current_lead_time)
    
    @property
    def state(self) -> DashboardState:
        return DashboardState(
            self.evaluation_filter.value,
            DOMAIN_MAPPING[self.domain_filter.value],
            DOMAIN_CONFIGURATION_MAPPING[self.domain_filter.value][self.configuration_filter.value],
            self.threshold_filter.value,
            self.metric_filter.value,
            self.confidence_filter.value,
            self.current_lead_time
        )

def generate_dashboard(
        root: Path,
        title: str
        ) -> BootstrapTemplate:
    return Dashboard(root, title).template
    # Data
    # NOTE
    # It will probably be easier to split up metric results into single
    # files that contain all the information needed.
    # Key available configurations to domain selection
    # Key evaluation metric retrieval to:
    #  configuration
    #  metric_name
    #  metric_value
    #  threshold_name
    #  threshold_value
    #  lead_time_min_hours (slider)
    #  confidence_interval_estimate (lower, central, upper)
    #  evaluation_label (e.g. 'FY2024 Q1 ROE')
    #  start_date
    #  end_date
    # routelink_reader = RoutelinkReader(root)
    # metrics_reader = MetricsReader(root)
    # domain_list = routelink_reader.domains
    # metric_labels = metrics_reader.metrics
    # configurations = metrics_reader.configurations
    # periods = metrics_reader.periods

    # # Plotters
    # site_map = SiteMapPlotter()

    # # Widgets
    # domain_selector = pn.widgets.Select(
    #     name="Select Domain",
    #     options=domain_list,
    #     value=domain_list[0]
    # )
    # configuration_selector = pn.widgets.Select(
    #     name="Select Model Configuration",
    #     options=configurations,
    #     value=configurations[0]
    # )
    # period_selector = pn.widgets.Select(
    #     name="Select Evaluation Period",
    #     options=periods,
    #     value=periods[0]
    # )
    # metric_selector = pn.widgets.Select(
    #     name="Select Metric",
    #     options=metric_labels,
    #     value=metric_labels[0]
    # )

    # # Panes
    # rng = np.random.default_rng(seed=2025)
    # geometry = routelink_reader.geometry(domain_list[0])
    # site_map.update_points(
    #     domain=domain_list[0],
    #     values=rng.uniform(-1.0, 1.0, len(geometry)),
    #     metric_label=metric_selector.value,
    #     routelink_reader=routelink_reader
    # )
    # site_map_pane = pn.pane.Plotly(site_map.figure)

    # # Callbacks
    # def domain_callbacks(domain):
    #     geometry = routelink_reader.geometry(domain)
    #     # Update map
    #     lat, lon, zoom = site_map.update_points(
    #         domain=domain,
    #         values=rng.uniform(-1.0, 1.0, len(geometry)),
    #         metric_label=metric_selector.value,
    #         routelink_reader=routelink_reader
    #     )
    #     site_map_pane.relayout_data.update({
    #         "map.center": {"lat": lat, "lon": lon}})
    #     site_map_pane.relayout_data.update({"map.zoom": zoom})
    #     site_map_pane.object = site_map.figure
    # pn.bind(domain_callbacks, domain_selector, watch=True)

    # def metric_callbacks(metric_label):
    #     # Update map
    #     site_map.update_colors(
    #         values=rng.uniform(-1.0, 1.0, len(geometry)),
    #         metric_label=metric_label,
    #         relayout_data=site_map_pane.relayout_data
    #     )
    #     site_map_pane.object = site_map.figure
    # pn.bind(metric_callbacks, metric_selector, watch=True)

    # # Layout
    # template = BootstrapTemplate(title=title)
    # template.sidebar.append(domain_selector)
    # template.sidebar.append(configuration_selector)
    # template.sidebar.append(period_selector)
    # template.sidebar.append(metric_selector)
    # template.main.append(site_map_pane)

    # Widgets
    # site_map = pn.pane.Plotly()
    # site_map_card = pn.Card(site_map, hide_header=True)

    # domain_filter = pn.widgets.Select(
    #     name="Model Domain",
    #     options=[
    #         "Alaska",
    #         "CONUS",
    #         "Hawaii",
    #         "Puerto Rico"
    #     ],
    #     value="Alaska"
    # )
    # threshold_filter = pn.widgets.Select(
    #     name="Streamflow Threshold",
    #     options=[
    #         "0th Percentile (All data)",
    #         "85th Percentile",
    #         "95th Percentile",
    #         "99th Percentile"
    #     ]
    # )
    # current_domain = DOMAIN_MAPPING[domain_filter.value]
    # configuration_filter = pn.widgets.Select(
    #     name="Model Configuration",
    #     options=list(DOMAIN_CONFIGURATION_MAPPING[current_domain].keys())
    #     )
    # metric_filter = pn.widgets.Select(
    #     name="Evaluation Metric",
    #     options=[
    #         "Mean relative bias",
    #         "Nash-Sutcliffe efficiency",
    #         "Pearson correlation coefficient",
    #         "Kling-Gupta efficiency"
    #     ]
    # )
    # filter_card = pn.Card(
    #     pn.Column(
    #         domain_filter,
    #         threshold_filter,
    #         configuration_filter,
    #         metric_filter
    #         ),
    #     title="Filters",
    #     collapsible=False
    #     )
    
    # # Callbacks
    # def handle_domain(domain):
    #     # Update configuration options
    #     current_domain = DOMAIN_MAPPING[domain]
    #     configuration_filter.options = list(
    #         DOMAIN_CONFIGURATION_MAPPING[current_domain].keys())
    # pn.bind(handle_domain, domain_filter, watch=True)

    # # Layout
    # grid = pn.Row(
    #     filter_card,
    #     site_map_card
    #     )
    # template = BootstrapTemplate(title=title)
    # template.main.append(grid)

    # return template

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
