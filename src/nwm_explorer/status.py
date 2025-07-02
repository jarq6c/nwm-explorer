"""Dashboard status alerts."""
from typing import Literal
import panel as pn

class StatusFeed:
    def __init__(self, title: str = "Status"):
        self.feed = pn.Feed(
            width=320,
            height=200,
            view_latest=False
            )
        self.card = pn.Card(
            self.feed,
            title=title,
            collapsible=False
            )
    
    def servable(self) -> pn.Card:
        return self.card
    
    def alert(
            self,
            message: str,
            alert_type: Literal[
                "primary",
                "secondary",
                "success",
                "danger",
                "warning",
                "info",
                "light",
                "dark"
                ]
        ) -> None:
        self.feed.append(pn.pane.Alert(message, alert_type=alert_type))
    
    def success(self, message: str) -> None:
        self.alert(message, "success")
    
    def danger(self, message: str) -> None:
        self.alert(message, "danger")
    
    def warning(self, message: str) -> None:
        self.alert(message, "warning")
    
    def info(self, message: str) -> None:
        self.alert(message, "info")
