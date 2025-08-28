"""
Dashboard layout.
"""
import inspect
from pathlib import Path
from typing import Any

import panel as pn
from panel.viewable import Viewer
from panel.template import BootstrapTemplate

from nwm_explorer.logging.loggers import get_logger
from nwm_explorer.layouts.post_event_evaluations import PostEventEvaluationLayout
from nwm_explorer.layouts.routine_operational_evaluations import RoutineOperationalEvaluationLayout
from nwm_explorer.panes.configuration import ConfigurationPane

class Dashboard(Viewer):
    """
    A dashboard for exploring National Water Model output.

    Parameters
    ----------
    root: Path
        Path to root data directory used by dashboards.
    title: str
        Title that appears in the dashboard header.
    params: any
        Additional keyword arguments passed directly to panel.viewable.Viewer.
    """
    def __init__(self, root: Path, title: str, **params: dict[str, Any]):
        # Apply parameters
        super().__init__(**params)

        # Get logger
        name = __loader__.name + "." + inspect.currentframe().f_code.co_name
        logger = get_logger(name)

        # Setup template
        logger.info("Build template")
        self.template = BootstrapTemplate(
            title=title,
            collapsed_sidebar=True,
            sidebar_width=380
        )

        # Dashboard modes
        self.modes = {
            "Routine Operational": RoutineOperationalEvaluationLayout(),
            "Post-event": PostEventEvaluationLayout()
        }
        self.template.main.append(pn.pane.Placeholder(
            object=list(self.modes.values())[0]
        ))

        # Dashboard configuration
        self.configuration = ConfigurationPane(
            mode_options=list(self.modes.keys()),
            streamflow_options=["foot^3/s", "m^3/s", "foot^3/s/mile^2", "inch/h", "mm/h"],
            precipitation_options=["inch/h", "mm/h"],
            map_layers=["Metrics", "USGS streamflow gages", "National Inventory of Dams"]
        )
        self.template.sidebar.append(self.configuration)

        # Update main area
        def change_dashboard_mode(mode_key) -> None:
            self.template.main[0].object = self.modes[mode_key]
        pn.bind(
            change_dashboard_mode,
            self.configuration.mode_selector.param.value,
            watch=True
            )

    def __panel__(self) -> BootstrapTemplate:
        return self.template

def generate_dashboard(
        root: Path,
        title: str
        ) -> Dashboard:
    """
    Generate a new dashboard. This insures that each user receives their own
    dashboard independent of other users.

    Parameters
    ----------
    root: Path
        Path to root data directory used by dashboards.
    title: str
        Title that appears in the dashboard header.
    
    Returns
    -------
    Dashboard
    """
    # Get logger
    name = __loader__.name + "." + inspect.currentframe().f_code.co_name
    logger = get_logger(name)
    logger.info("Build dashboard")
    return Dashboard(root, title)
