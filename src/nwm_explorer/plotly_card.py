from dataclasses import dataclass
from typing import Any
import panel as pn
import plotly.graph_objects as go
from plotly.basedatatypes import BaseTraceType

@dataclass
class PlotlyCard:
    data: list[BaseTraceType]
    layout: go.Layout

    def __post_init__(self) -> None:
        # Build pane
        self.pane = pn.pane.Plotly({
            "data": self.data,
            "layout": self.layout
        }, config={'displaylogo': False})

        # Build card
        self.card = pn.Card(
            self.pane,
            collapsible=False,
            hide_header=True
        )
    
    def refresh(self) -> None:
        # Update pane
        self.pane.object = {
            "data": self.data,
            "layout": self.layout
        }
    
    def update_layout(self, parameters: dict[str, Any]) -> None:
        # Update layout
        self.layout.update(parameters)
    
    def update_data(self, parameters: dict[str, Any], index: int = 0) -> None:
        # Update trace
        self.data[index].update(parameters)

    def replace_data(self, data: list[BaseTraceType]) -> None:
        # Update trace
        self.data = data

    def update_config(self, parameters: dict[str, Any]) -> None:
        # Assign config is non-existent
        if self.pane.config is None:
            self.pane.config = parameters
            return
        
        # Update existing config
        self.pane.config.update(parameters)
    
    def update_xaxis_title_text(self, title: str) -> None:
        # Update layout
        self.layout["xaxis"]["title"].update({"text": title})
    
    def update_yaxis_title_text(self, title: str) -> None:
        # Update layout
        self.layout["yaxis"]["title"].update({"text": title})
    
    def update_xaxis_range(self, xrange: list[float]) -> None:
        # Update layout
        self.layout["xaxis"].update({"range": xrange})
    
    def update_yaxis_range(self, yrange: list[float]) -> None:
        # Update layout
        self.layout["yaxis"].update({"range": yrange})
    
    def update_line(self, parameters: dict[str, Any], index: int = 0) -> None:
        # Update trace
        self.data[index]["line"].update(parameters)

        # Retain layout
        if "xaxis.range[0]" in self.pane.relayout_data:
            rdata = self.pane.relayout_data
            xrange = [rdata["xaxis.range[0]"], rdata["xaxis.range[1]"]]
            yrange = [rdata["yaxis.range[0]"], rdata["yaxis.range[1]"]]
            self.update_xaxis_range(xrange)
            self.update_yaxis_range(yrange)
        else:
            self.update_xaxis_range(None)
            self.update_yaxis_range(None)
    
    def servable(self) -> pn.Card:
        return self.card
