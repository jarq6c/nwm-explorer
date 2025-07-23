"""Generate standardized hydrograph plots."""
import numpy.typing as npt
import panel as pn
import colorcet as cc
import plotly.graph_objects as go
from plotly.colors import hex_to_rgb, label_rgb

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

class Hydrograph:
    def __init__(self):
        self.data = [go.Scatter()]
        self.default_layout = go.Layout(
            height=250,
            width=1045,
            margin=dict(l=0, r=0, t=0, b=0),
            clickmode="event",
            yaxis=dict(title=dict(text="Streamflow (cfs)"))
        )
        self.layout = self.default_layout
        self.figure = {
            "data": self.data,
            "layout": self.layout
        }
        self.pane = pn.pane.Plotly(self.figure)

        # Trace highlighting
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
                self.data[self.curve_number]["line"].update(dict(
                    color=self.curve_color,
                    width=self.curve_width
                ))

            # Update current curve
            self.curve_number = event["points"][0]["curveNumber"]
            trace = self.data[self.curve_number]
            if "lines" not in trace["mode"]:
                return
            self.curve_color = trace["line"]["color"]
            self.curve_width = trace["line"]["width"]

            # Invert colors
            self.data[self.curve_number]["line"].update(dict(
                color=invert_color(self.curve_color),
                width=self.curve_width+4
            ))
            self.refresh()
        pn.bind(highlight_trace, self.pane.param.click_data, watch=True)

        def maintain_focus(event):
            if event is None:
                return
            
            # Reset layout
            self.layout = self.default_layout
            
            # Update layout to reflect zoom
            if "xaxis.range[0]" in event:
                self.layout["xaxis"].update(dict(
                    range=[event["xaxis.range[0]"], event["xaxis.range[1]"]]
                ))
                self.layout["yaxis"].update(dict(
                    range=[event["yaxis.range[0]"], event["yaxis.range[1]"]]
                ))
        pn.bind(maintain_focus, self.pane.param.relayout_data, watch=True)
    
    def update_data(
            self, 
            x: list[npt.ArrayLike],
            y: list[npt.ArrayLike],
            names: list[str]
        ) -> None:
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

        # Trace highlighting
        self.curve_number = None
        self.curve_color = None
        self.curve_width = None

        # Update data
        self.data = data
        self.layout = self.default_layout
    
    def refresh(self) -> None:
        self.figure.update(dict(data=self.data, layout=self.layout))
        self.pane.object = self.figure
    
    def servable(self) -> pn.Card:
        return pn.Card(
            self.pane,
            collapsible=False,
            hide_header=True
        )
