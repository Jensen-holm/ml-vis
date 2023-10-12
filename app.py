from flask import Flask, request, Response, jsonify
from flask_cors import CORS

import pandas as pd

from plot.loss_hist import loss_hist

app = Flask(__name__)

CORS(app, origins="*")


@app.route("/neural-network", methods=["POST"])
def nn_plot():
    data = request.json
    try:
        loss_hist_epoch = loss_hist(
            loss_hist_epoch=data["loss_hist"],
        )
        return Response(
            loss_hist_epoch,
            status=200,
            content_type="image/svg",
        )
    except Exception as e:
        return Response(f"error plotting loss_hist plot: {e}", status=400)


if __name__ == "__main__":
    app.run(
        port=3000,
        debug=True,
    )
