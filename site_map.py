"""Site map card."""
import numpy as np
import numpy.typing as npt
import polars as pl
import panel as pn
import plotly.graph_objects as go
import colorcet as cc
from nwm_explorer.plotly_card import PlotlyCard

class SiteMapCard:
    def __init__(
            self,
            latitude: npt.ArrayLike,
            longitude: npt.ArrayLike,
            custom_data: pl.DataFrame,
            values: npt.ArrayLike,
            value_label: str,
            value_limits: tuple[float, float],
            custom_labels: list[str],
            default_zoom: int
            ):
        # Hover template
        hover_template = ""
        for idx, c in enumerate(custom_labels):
            hover_template += f"{c}: %" + "{customdata[" + str(idx) + "]}<br>"

        # Make map
        data = [go.Scattermap(
            marker=dict(
                color=values,
                colorbar=dict(
                    title=dict(
                        text=value_label,
                        side="right"
                        )
                    ),
                cmin=value_limits[0],
                cmax=value_limits[1],
                size=15,
                colorscale=cc.gouldian
            ),
            lat=latitude,
            lon=longitude,
            customdata=custom_data,
            hovertemplate=(
                hover_template +
                "Longitude: %{lon}<br>"
                "Latitude: %{lat}"
                f"<br>{value_label}: "
                "%{marker.color:.2f}"
            ),
            showlegend=False,
            name="",
            mode="markers"
        )]

        # Create card
        self.card = PlotlyCard(
            data=data,
            layout=go.Layout(
                showlegend=False,
                height=540,
                width=850,
                margin=dict(l=0, r=0, t=0, b=0),
                map=dict(
                    style="satellite-streets",
                    center=dict(
                        lat=np.mean(latitude),
                        lon=np.mean(longitude)
                    ),
                    zoom=default_zoom
                ),
                clickmode="event",
                modebar=dict(
                    remove=["lasso", "select"],
                    orientation="v"
                ),
                dragmode="zoom",
            )
        )
    
    def servable(self) -> pn.Card:
        return self.card.servable()
    
    def refresh(self) -> None:
        self.card.refresh()

def main():
    latitude = [61.2255]
    longitude = [-149.6429]
    custom_data = pl.DataFrame({
        "usgs_site_code": ["1527600"],
        "nwm_feature_id": [75000100003094]
    })
    values = [0.52]
    value_label = "Kling-Gupta Efficiency"
    value_limits = [-1.0, 1.0]
    custom_labels = ["USGS Site Code", "NWM Feature ID"]
    default_zoom = 5

    card = SiteMapCard(
        latitude=latitude,
        longitude=longitude,
        custom_data=custom_data,
        values=values,
        value_label=value_label,
        value_limits=value_limits,
        custom_labels=custom_labels,
        default_zoom=default_zoom
    )

    pn.serve(card.servable())

if __name__ == "__main__":
    main()
