from dataclasses import dataclass
from typing import Any
import numpy as np
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
        })

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

def main():
    x = np.arange(10)
    y = 1.0 * np.exp(-0.5 * x)
    y_lo = y * 0.8
    y_hi = y * 1.2
    y_lo_err = y - y_lo
    y_hi_err = y_hi - y
    l = [f"{n:.0f}" for n in x]
    xlabel = "Minimum Lead Time (h)"
    ylabel = "Kling-Gupta Efficiency"
    custom_data = np.hstack((y_lo[:, np.newaxis], y_hi[:, np.newaxis]))

    data = [go.Bar(
        x=l,
        y=y,
        customdata=custom_data,
        hovertemplate=(
            f"{xlabel}: " + "%{x}<br>" + 
            f"{ylabel}: " + "%{customdata[0]:.2f} -- %{customdata[1]:.2f} (%{y:.2f})"
        ),
        name="",
        error_y=dict(
            type="data",
            array=y_hi_err,
            arrayminus=y_lo_err
        )
    )]
    layout = go.Layout(
        height=250,
        width=250,
        xaxis=dict(title=dict(text=xlabel)),
        yaxis=dict(title=dict(text=ylabel)),
        margin=dict(l=0, r=0, t=0, b=0)
    )
    card = PlotlyCard(data, layout)
    pn.serve(card.servable())

if __name__ == "__main__":
    main()
