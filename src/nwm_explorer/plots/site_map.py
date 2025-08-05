"""Plotting."""
import numpy.typing as npt
import pandas as pd
import panel as pn
import plotly.graph_objects as go
from plotly.basedatatypes import BaseTraceType
import colorcet as cc

class SiteMap:
    def __init__(
            self,
            center: dict[str, float],
            zoom: float,
            additional_layers: dict[str, BaseTraceType] | None = None
            ):
        # Extra layers
        self.additional_layers = additional_layers

        # Map data
        self.data = [go.Scattermap(
            marker=dict(
                colorbar=dict(
                    title=dict(
                        side="right"
                        )
                    ),
                size=15,
                colorscale=cc.gouldian
            ),
            showlegend=False,
            name="",
            mode="markers"
        )]

        # Map layout
        self.layout = go.Layout(
            showlegend=False,
            height=540,
            width=850,
            margin=dict(l=0, r=0, t=0, b=0),
            map=dict(
                style="satellite-streets",
                center=center,
                zoom=zoom
            ),
            clickmode="event",
            modebar=dict(
                remove=["lasso", "select", "resetview"],
                orientation="v"
            ),
            dragmode="zoom"
        )

        # Map figure
        self.figure = dict(
            data=self.data+list(self.additional_layers.values()),
            layout=self.layout
        )

        # Servable
        self.pane = pn.pane.Plotly(self.figure)
    
    def update(
        self,
        values: npt.ArrayLike,
        latitude: npt.ArrayLike,
        longitude: npt.ArrayLike,
        value_label: str,
        cmin: float,
        cmax: float,
        custom_data: pd.DataFrame
        ) -> None:
        # Colors
        self.data[0]["marker"].update(dict(color=values, cmin=cmin, cmax=cmax))

        # ScatterMap
        self.data[0].update(dict(
            lat=latitude,
            lon=longitude,
            customdata=custom_data,
            hovertemplate=(
                f"<br>{value_label}: "
                "%{marker.color:.2f}<br>"
                "NWM Feature ID: %{customdata[0]}<br>"
                "USGS Site Code: %{customdata[1]}<br>"
                "Start Date: %{customdata[2]}<br>"
                "End Date: %{customdata[3]}<br>"
                "Samples: %{customdata[4]}<br>"
                "Longitude: %{lon}<br>"
                "Latitude: %{lat}"
        )))

        # Title
        self.data[0]["marker"]["colorbar"]["title"].update(dict(text=value_label))
    
    def refresh(self) -> None:
        self.figure.update(dict(
            data=self.data+list(self.additional_layers.values()),
            layout=self.layout
        ))
        self.pane.object = self.figure
    
    def servable(self) -> pn.Card:
        return pn.Card(
            self.pane,
            collapsible=False,
            hide_header=True
        )
