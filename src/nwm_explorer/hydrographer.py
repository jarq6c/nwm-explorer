"""Generate standardized histogram plots."""
import numpy.typing as npt
import panel as pn
import colorcet as cc
import plotly.graph_objects as go
from plotly.colors import hex_to_rgb, label_rgb
from nwm_explorer.plotly_card import PlotlyCard

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
                height=250,
                width=1045,
                margin=dict(l=0, r=0, t=0, b=0),
                clickmode="event"
                )
        )

        # Set y-label
        self.card.update_yaxis_title_text(y_title)

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
            self.refresh()
        pn.bind(highlight_trace, self.click_data, watch=True)
    
    @property
    def click_data(self) -> dict:
        return self.card.click_data
    
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

        # Update interface
        self.card.data = data
    
    def servable(self) -> pn.Card:
        return self.card.servable()
    
    def refresh(self) -> None:
        self.card.refresh()
