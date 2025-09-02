"""
Primary interface for the National Water Model Evaluation Explorer.

This utility includes commands and sub-commands used to build and manage a
data store of predictions, observations, evaluations, and supporting
metadata; export data to CSV; and launch a web-based GUI.
"""
from pathlib import Path

import click

from nwm_explorer.application.gui import serve_dashboards
from nwm_explorer.application.api import EvaluationRegistry

# Register sub-commands
display_group = click.Group()

@display_group.command()
@click.option("-r", "--registry", "registry", nargs=1, type=click.Path(path_type=Path), default="evaluation_registry.json",
    help="Path to evaluation registry (./evaluation_registry.json)")
def display(
    registry: Path = Path("evaluation_registry.json")
    ) -> None:
    """Visualize and explore evaluation data.

    Example:
    
    nwm-explorer display
    """
    # Validate and load registry
    with registry.open("r") as fi:
        evaluation_registry = EvaluationRegistry.model_validate_json(fi.read())
    serve_dashboards(evaluation_registry)

# Package commands under a single CLI
cli = click.CommandCollection(sources=[
    display_group
    ])

# Run this script to interact with the CLI
if __name__ == "__main__":
    cli()
