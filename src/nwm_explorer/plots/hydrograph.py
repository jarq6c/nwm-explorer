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
        self.layout = go.Layout(
            height=250,
            width=1045,
            margin=dict(l=0, r=0, t=0, b=0),
            clickmode="event",
            yaxis=dict(title=dict(text="Streamflow (cfs)"))
        )
        self.figure = {
            "data": self.data,
            "layout": self.layout
        }
        self.pane = pn.pane.Plotly(self.figure)
    
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

        # Update data
        self.data = data
    
    def refresh(self) -> None:
        self.figure.update(dict(data=self.data, layout=self.layout))
        self.pane.object = self.figure
    
    def servable(self) -> pn.Card:
        return pn.Card(
            self.pane,
            collapsible=False,
            hide_header=True
        )
