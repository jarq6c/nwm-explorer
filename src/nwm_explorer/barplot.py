"""Display standardized bar plots."""
import numpy as np
import numpy.typing as npt
import panel as pn
import plotly.graph_objects as go
from nwm_explorer.plotly_card import PlotlyCard

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
            width=440,
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

        # Update axis titles
        self.card.update_xaxis_title_text(xlabel)
        self.card.update_yaxis_title_text(ylabel)
    
    def servable(self) -> pn.Card:
        return self.card.servable()
    
    def refresh(self) -> None:
        self.card.refresh()
