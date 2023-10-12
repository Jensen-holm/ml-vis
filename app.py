from flask import Flask, Response, request, jsonify
from flask_cors import CORS

from plot.loss_hist import loss_hist

app = Flask(__name__)


@app.route("/neural-network", methods=["POST"])
def nn_plot():
    data = request.json

    fig_bytes = loss_hist(data["loss_hist"])
    response = Response(fig_bytes, content_type='image/png')
    return response


if __name__ == "__main__":
    app.run(
        port=3000,
        debug=True,
    )
