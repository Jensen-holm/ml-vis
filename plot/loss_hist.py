import plotly.graph_objects as go
import plotly.io as pio
import numpy as np

x = np.arange(10)


def loss_hist(loss_hist_epoch):
    fig = go.Figure(
        data=go.Scatter(
            x=np.arange(len(loss_hist_epoch)),
            y=loss_hist_epoch,
            mode="lines",
            name="epoch, log loss",
            line_color='rgb(0,176,246)',
        )
    )

    fig.update_layout(
        title='Log Loss / Epoch',
        xaxis_title='Epoch',
        yaxis_title='Log Loss',
    )

    return pio.to_image(fig, format='svg')
