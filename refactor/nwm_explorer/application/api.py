"""
Objects and methods needed to support programmatic interaction with various
application components.
"""
from pydantic import BaseModel

class DashboardConfiguration(BaseModel):
    """
    Dashboard configuration options.

    Attributes
    ----------
    title: str
        Dashboard title displayed on header.
    slug: str
        Dashboard slug appended to hostname in the URL.
    """
    title: str
    slug: str

class EvaluationRegistry(BaseModel):
    """
    Centralized object used to track various options used through the application.

    Attributes
    ----------
    dashboard_configuration: DashboardConfiguration
        Dashboard configuration options.
    """
    dashboard_configuration: DashboardConfiguration
