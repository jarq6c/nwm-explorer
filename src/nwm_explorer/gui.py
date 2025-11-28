"""Basic functions to build and launch dashboards."""
from pathlib import Path

import panel as pn

from nwm_explorer.dashboard import Dashboard
from nwm_explorer.configuration import load_configuration, Configuration

def generate_dashboard(
        configuration: Configuration
        ) -> Dashboard:
    """Instantiate and return a dashboard."""
    return Dashboard(configuration)

def generate_dashboard_closure(
        configuration: Configuration
        ) -> Dashboard:
    """Build and return a closure function that generates new dashboards."""
    def closure():
        return generate_dashboard(configuration)
    return closure

def serve_dashboards(
        configuration_file: Path
        ) -> None:
    """Serve dashboards."""
    # Load configuration
    configuration = load_configuration(configuration_file)

    # Serve
    endpoints = {
        configuration.endpoint: generate_dashboard_closure(configuration)
    }
    pn.serve(endpoints)
