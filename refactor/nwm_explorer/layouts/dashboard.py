"""
Dashboard layout.
"""
import inspect
from pathlib import Path
from typing import Any

from panel.viewable import Viewer
from panel.template import BootstrapTemplate

from nwm_explorer.logging.loggers import get_logger
from nwm_explorer.layouts.post_event_evaluations import PostEventEvaluationLayout
from nwm_explorer.layouts.routine_operational_evaluations import RoutineOperationalEvaluationLayout

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
            collapsed_sidebar=True
        )

        # Add post-event evaluations
        self.template.main.append(PostEventEvaluationLayout())
        self.template.main.append(RoutineOperationalEvaluationLayout())

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
