from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np


def plot(loss_hist_epoch, accuracy_scores):

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "Log Loss / Epoch",
            "Accuracy Score / Epoch",
        ])
    # Update x-axis and y-axis tick font color to white
    fig.update_xaxes(tickfont=dict(color="white"))
    fig.update_yaxes(tickfont=dict(color="white"))

    fig.update_layout(
        title_font=dict(color="white"),
    )

    trace1 = go.Scatter(
        x=np.arange(len(loss_hist_epoch)),
        y=loss_hist_epoch,
        mode="lines",
        name="epoch, log loss",
        # line_color='rgb(13, 133, 85, 0)',
        showlegend=False,
    )

    trace2 = go.Scatter(
        x=np.arange(len(accuracy_scores)),
        y=accuracy_scores,
        mode="lines",
        name="accuracy scores / epoch",
        line_color='rgb(199, 42, 47, 0)',
        showlegend=False,
    )

    # Add the traces to the subplots
    fig.add_trace(trace1, row=1, col=1)
    fig.add_trace(trace2, row=1, col=2)

    fig.update_layout(
        title_text='Classification Results',
        paper_bgcolor="rgba(0, 0, 0, 0)",
        width=1000,
    )
    return pio.to_image(fig, format="png")
