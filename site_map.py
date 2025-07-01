"""Site map card."""
from typing import Any
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
        self.hover_template = ""
        for idx, c in enumerate(custom_labels):
            self.hover_template += f"{c}: %" + "{customdata[" + str(idx) + "]}<br>"

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
                self.hover_template +
                "Longitude: %{lon}<br>"
                "Latitude: %{lat}"
                f"<br>{value_label}: "
                "%{marker.color:.2f}"
            ),
            showlegend=False,
            name="",
            mode="markers"
        )]

        # Default map options
        self.map_center = {
            "lon": np.mean(longitude),
            "lat": np.mean(latitude)
        }
        self.map_zoom = default_zoom

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
                    center=self.map_center,
                    zoom=self.map_zoom
                ),
                clickmode="event",
                modebar=dict(
                    remove=["lasso", "select"],
                    orientation="v"
                ),
                dragmode="zoom"
            )
        )

        # Boundaries
        self.lat_min = np.min(latitude)
        self.lat_max = np.max(latitude)
        self.lon_min = np.min(longitude)
        self.lon_max = np.max(longitude)
        pn.bind(self.update_boundaries, self.relayout_data, watch=True)

        # Click data
        self.columns = custom_data.columns
        self.selection: dict[str, Any] = {}
        pn.bind(self.update_selection, self.click_data, watch=True)
    
    @property
    def click_data(self) -> dict:
        return self.card.click_data
    
    @property
    def relayout_data(self) -> dict:
        return self.card.relayout_data
    
    def update_boundaries(self, data: dict) -> None:
        if "map._derived" in data:
            self.lat_max = data["map._derived"]["coordinates"][0][1]
            self.lat_min = data["map._derived"]["coordinates"][2][1]
            self.lon_max = data["map._derived"]["coordinates"][1][0]
            self.lon_min = data["map._derived"]["coordinates"][0][0]
            self.map_center = data["map.center"]
            self.map_zoom = data["map.zoom"]
        elif "map.center" in data:
            self.lat_min = np.min(self.card.data[0].lat)
            self.lat_max = np.max(self.card.data[0].lat)
            self.lon_min = np.min(self.card.data[0].lon)
            self.lon_max = np.max(self.card.data[0].lon)
            self.map_center = data["map.center"]
            self.map_zoom = data["map.zoom"]

        # Re-center map
        self.card.recenter_map(dict(
            center=self.map_center,
            zoom=self.map_zoom
        ))
        
    def update_selection(self, data: dict) -> None:
        custom_data = data["points"][0]["customdata"]
        for key, value in zip(self.columns, custom_data):
            self.selection[key] = value
        self.selection["value"] = data["points"][0]["marker.color"]
        self.selection["lon"] = data["points"][0]["lon"]
        self.selection["lat"] = data["points"][0]["lat"]
    
    def update_points(
            self,
            latitude: npt.ArrayLike,
            longitude: npt.ArrayLike,
            custom_data: pl.DataFrame,
            values: npt.ArrayLike,
            value_label: str,
            value_limits: tuple[float, float],
            custom_labels: list[str],
            default_zoom: int
        ) -> None:
        # Hover template
        self.hover_template = ""
        for idx, c in enumerate(custom_labels):
            self.hover_template += f"{c}: %" + "{customdata[" + str(idx) + "]}<br>"
        
        # Data
        self.card.update_data(dict(
            lat=latitude,
            lon=longitude,
            customdata=custom_data,
            hovertemplate=(
                self.hover_template +
                "Longitude: %{lon}<br>"
                "Latitude: %{lat}"
                f"<br>{value_label}: "
                "%{marker.color:.2f}"
            )
        ))

        # Markers
        self.card.update_markers(dict(
            color=values,
            colorbar=dict(
                title=dict(
                    text=value_label
                    )
                ),
            cmin=value_limits[0],
            cmax=value_limits[1]
        ))

        # Default map options
        self.map_center = {
            "lon": np.mean(longitude),
            "lat": np.mean(latitude)
        }
        self.map_zoom = default_zoom

        # Re-center map
        self.card.recenter_map(dict(
            center=self.map_center,
            zoom=self.map_zoom
        ))

        # Update boundaries
        self.lat_min = np.min(latitude)
        self.lat_max = np.max(latitude)
        self.lon_min = np.min(longitude)
        self.lon_max = np.max(longitude)

        # Click data
        self.columns = custom_data.columns
        self.selection = {}
    
    def update_values(
            self,
            values: npt.ArrayLike,
            value_label: str,
            value_limits: tuple[float, float]
        ) -> None:
        # Data
        self.card.update_data(dict(
            hovertemplate=(
                self.hover_template +
                "Longitude: %{lon}<br>"
                "Latitude: %{lat}"
                f"<br>{value_label}: "
                "%{marker.color:.2f}"
            )
        ))

        # Markers
        self.card.update_markers(dict(
            color=values,
            colorbar=dict(
                title=dict(
                    text=value_label
                    )
                ),
            cmin=value_limits[0],
            cmax=value_limits[1]
        ))

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
    def update_map(event):
        print(event)
        print(card.lat_max)
        print(card.lat_min)
        print(card.lon_max)
        print(card.lon_min)
        print(card.selection)
    pn.bind(update_map, card.click_data, watch=True)

    latitude = [37.0042]
    longitude = [-95.59124]
    custom_data = pl.DataFrame({
        "usgs_site_code": ["07170990"],
        "nwm_feature_id": [21797449]
    })
    values = [0.44]
    value_label = "Mean Relative Bias"
    value_limits = [0.0, 2.0]
    custom_labels = ["USGS Site Code", "NWM Feature ID"]
    default_zoom = 3

    card.update_points(
        latitude=latitude,
        longitude=longitude,
        custom_data=custom_data,
        values=values,
        value_label=value_label,
        value_limits=value_limits,
        custom_labels=custom_labels,
        default_zoom=default_zoom
    )
    card.refresh()

    rng = np.random.default_rng()
    def update_colors(event):
        if not event:
            return
        gain = rng.uniform(0.0, 1.0)
        card.update_values([values[0]*gain], f"GAIN ({gain:.2f})", (0.0, gain))
        card.refresh()
    button = pn.widgets.Button(name="Click me", button_type="primary")
    pn.bind(update_colors, button, watch=True)

    pn.serve(pn.Column(card.servable(), button))

if __name__ == "__main__":
    main()
