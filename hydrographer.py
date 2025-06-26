"""Generate standardized histogram plots."""
from dataclasses import dataclass
from typing import Any
import numpy as np
import numpy.typing as npt
import panel as pn
import colorcet as cc
import plotly.graph_objects as go
from plotly.colors import hex_to_rgb, label_rgb
from plotly.basedatatypes import BaseTraceType

def invert_color(value: str) -> str:
    """Convert a hex color to an inverted rgb label.
    
    Parameters
    ----------
    value: str, required,
        Hex color string.
    
    Returns
    -------
    str:
        Inverted rgb color.
    """
    r, g, b = hex_to_rgb(value)
    return label_rgb((255-r, 255-g, 255-b))

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

class HydrographCard:
    def __init__(
            self,
            x: list[npt.ArrayLike],
            y: list[npt.ArrayLike],
            names: list[str],
            y_title: str
            ):
        # Assume first trace is special
        data = [go.Scatter(
            x=x[0],
            y=y[0],
            mode="lines",
            line=dict(color="#3C00FF", width=2),
            name=names[0]
        )]

        # Generate remaining traces
        color_index = 0
        for idx in range(1, len(y)):
            data.append(go.Scatter(
                x=x[idx],
                y=y[idx],
                mode="lines",
                name=names[idx],
                line=dict(color=cc.CET_L8[color_index], width=1)
                ))
            color_index += 1
            if color_index == len(cc.CET_L8):
                color_index = 0

        # Create card
        self.card = PlotlyCard(
            data=data,
            layout=go.Layout(
                height=300,
                width=800,
                margin=dict(l=0, r=0, t=0, b=0),
                clickmode="event"
                )
        )

        # Set y-label
        self.card.update_yaxis_title_text(y_title)
        self.card.refresh()

        self.curve_number = None
        self.curve_color = None
        self.curve_width = None
        def highlight_trace(event):
            if not event:
                return
            if event["points"][0]["curveNumber"] == self.curve_number:
                return
            
            # Restore color of old line
            if self.curve_number is not None:
                self.card.update_line(
                    dict(color=self.curve_color, width=self.curve_width),
                    self.curve_number
                )

            # Update current curve
            self.curve_number = event["points"][0]["curveNumber"]
            trace = self.card.data[self.curve_number]
            if "lines" not in trace["mode"]:
                return
            self.curve_color = trace["line"]["color"]
            self.curve_width = trace["line"]["width"]

            # Invert colors
            self.card.update_line(
                dict(
                    color=invert_color(self.curve_color),
                    width=self.curve_width+4
                    ),
                self.curve_number
            )
            self.card.refresh()
        pn.bind(highlight_trace, self.card.pane.param.click_data, watch=True)
    
    def servable(self) -> pn.Card:
        return self.card.servable()

def main():
    N = 100
    x = [np.linspace(0, 10, 50)+idx for idx in range(N)]
    y = [0.5 * x[idx] + idx for idx in range(N)]
    card = HydrographCard(
        x=x,
        y=y,
        names=["USGS-01013500"]+[f"Forecast {idx}" for idx in range(len(y))],
        y_title="STREAMFLOW (CFS)"
    )
    pn.serve(card.servable())

if __name__ == "__main__":
    main()
