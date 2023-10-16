import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import numpy as np


def plot(loss_hist_epoch, accuracy_score):

    # Create a subplot with 1 row and 2 columns
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Log Loss / Epoch",
                        "Accuracy Score on Validation Data"],
        specs=[[{"type": "xy"}, {"type": "domain"}]],
    )

    # Create the log loss / epoch plot
    fig.add_trace(go.Scatter(
        x=np.arange(len(loss_hist_epoch)),
        y=loss_hist_epoch,
        mode="lines",
        name="epoch, log loss",
        showlegend=False,
    ), row=1, col=1)

    # Create a pie chart for accuracy score
    fig.add_trace(go.Pie(
        labels=["Correct", "Incorrect"],
        values=[accuracy_score, 1 - accuracy_score],
        showlegend=True,
        marker=dict(colors=['green', 'red']),
    ), row=1, col=2)

    fig.update_layout(
        title_text='Classification Results',
        paper_bgcolor="rgba(0, 0, 0, 0)",
        width=1000,
    )

    return pio.to_image(fig, format="png")


# Example usage
loss_hist_epoch = [0.1, 0.08, 0.06, 0.04, 0.02]
accuracy_score = 0.85  # Replace with your actual accuracy score

plot(loss_hist_epoch, accuracy_score)
