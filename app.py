from flask import Flask, request, Response
from flask_cors import CORS

from plot.plot import plot

app = Flask(__name__)

CORS(app, origins="*")


@app.route("/neural-network", methods=["POST"])
def nn_plot():
    data = request.json
    try:
        data_vis = plot(
            loss_hist_epoch=data["loss_hist"],
            accuracy_scores=data["accuracy_scores"],
        )
        return Response(
            data_vis,
            status=200,
            content_type="image/png",
        )
    except Exception as e:
        return Response(f"error plotting loss_hist plot: {e}", status=400)


if __name__ == "__main__":
    app.run()
