from plotly.subplots import make_subplots
import plotly.graph_objs as go
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

    # Create the traces
    trace1 = go.Scatter(
        x=np.arange(len(loss_hist_epoch)),
        y=loss_hist_epoch,
        mode="lines",
        name="epoch, log loss",
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

    # Create frames for animation
    frames = []
    for i in range(len(loss_hist_epoch)):
        frame1 = go.Frame(data=[go.Scatter(x=np.arange(i+1), y=loss_hist_epoch[:i+1])],
                          name=f"Frame {i}")
        frame2 = go.Frame(data=[go.Scatter(x=np.arange(i+1), y=accuracy_scores[:i+1])],
                          name=f"Frame {i}")
        frames.extend([frame1, frame2])

    fig.frames = frames

    # Specify the duration for each frame (in milliseconds)
    frame_duration = 1000  # 1 second per frame

    # Update layout to play animation without user controls
    fig.update_layout(updatemenu=[], showlegend=False)
    fig.update(frames=[dict(duration=frame_duration, redraw=True)
               for _ in frames])

    fig.update_layout(
        title_text='Classification Results',
        paper_bgcolor="rgba(0, 0, 0, 0)",
        width=1000,
    )

    return pio.to_image(fig, format="svg")
