"""
Primary interface for the National Water Model Evaluation Explorer.

This utility includes commands and sub-commands used to build and manage a
data store of predictions, observations, evaluations, and supporting
metadata.
"""
from pathlib import Path

import click

from nwm_explorer.application.gui import serve_dashboard

# Register sub-commands
display_group = click.Group()

@display_group.command()
@click.option("-d", "--directory", "directory", nargs=1, type=click.Path(path_type=Path), default="data", help="Data directory (./data)")
@click.option("-t", "--title", "title", nargs=1, type=click.STRING, default="National Water Model Evaluations", help="Dashboard title header")
def display(
    directory: Path = Path("data"),
    title: str = "National Water Model Evaluations"
    ) -> None:
    """Visualize and explore evaluation data.

    Example:
    
    nwm-explorer display
    """
    serve_dashboard(directory, title)

cli = click.CommandCollection(sources=[
    display_group
    ])

if __name__ == "__main__":
    cli()
