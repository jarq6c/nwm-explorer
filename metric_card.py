from dataclasses import dataclass
from typing import Any
import numpy as np
import numpy.typing as npt
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
    
class BarPlot:
    def __init__(
            self,
            xdata: npt.ArrayLike,
            ydata: npt.ArrayLike,
            ydata_lower: npt.ArrayLike,
            ydata_upper: npt.ArrayLike,
            xlabel: str,
            ylabel: str
            ) -> None:
        custom_data = np.hstack((ydata_lower[:, np.newaxis], ydata_upper[:, np.newaxis]))
        data = [go.Bar(
            x=xdata,
            y=ydata,
            customdata=custom_data,
            hovertemplate=(
                f"{xlabel}: " + "%{x}<br>" + 
                f"{ylabel}: " + "%{customdata[0]:.2f} -- %{customdata[1]:.2f} (%{y:.2f})"
            ),
            name="",
            error_y=dict(
                type="data",
                array=ydata_upper - ydata,
                arrayminus=ydata - ydata_lower
            )
        )]
        layout = go.Layout(
            height=250,
            width=250,
            xaxis=dict(title=dict(text=xlabel)),
            yaxis=dict(title=dict(text=ylabel)),
            margin=dict(l=0, r=0, t=0, b=0),
            modebar=dict(
                remove=["lasso", "select", "pan", "autoscale", "zoomin", "zoomout"],
                orientation="v"
            )
        )

        self.card = PlotlyCard(data, layout)
    
    def update_data(
            self, 
            xdata: npt.ArrayLike,
            ydata: npt.ArrayLike,
            ydata_lower: npt.ArrayLike,
            ydata_upper: npt.ArrayLike,
            xlabel: str,
            ylabel: str
        ) -> None:
        # Construct custom data
        custom_data = np.hstack((ydata_lower[:, np.newaxis], ydata_upper[:, np.newaxis]))

        # Update trace
        self.card.update_data(dict(
            x=xdata,
            y=ydata,
            customdata=custom_data,
            hovertemplate=(
                f"{xlabel}: " + "%{x}<br>" + 
                f"{ylabel}: " + "%{customdata[0]:.2f} -- %{customdata[1]:.2f} (%{y:.2f})"
            ),
            error_y=dict(
                type="data",
                array=ydata_upper - ydata,
                arrayminus=ydata - ydata_lower
            )
        ))
    
    def servable(self) -> pn.Card:
        return self.card.servable()
    
    def refresh(self) -> None:
        self.card.refresh()

def main():
    x = np.arange(10)
    y = 1.0 * np.exp(-0.5 * x)

    barplot = BarPlot(
        xdata = [f"{n:.0f}" for n in x],
        ydata = y,
        ydata_lower = y * 0.8,
        ydata_upper = y * 1.2,
        xlabel = "Minimum Lead Time (h)",
        ylabel = "Kling-Gupta Efficiency"
    )
    x = np.arange(10)+24
    y = 1_000_000 * np.exp(-0.5 * x)

    barplot.update_data(
        xdata = [f"{n:.0f}" for n in x],
        ydata = y,
        ydata_lower = y * 0.8,
        ydata_upper = y * 1.2,
        xlabel = "Minimum Lead Time (h)",
        ylabel = "Relative Mean"
    )
    barplot.refresh()
    pn.serve(barplot.servable())

if __name__ == "__main__":
    main()
