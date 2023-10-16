import requests

with open("mushrooms.csv", "rb") as csv:
    data = csv.read()

headers = {
    "Content-Type": "application/json",
}

ARGS = {
    "epochs": 100,
    "hidden_size": 12,
    "learning_rate": 0.01,
    "test_size": 0.2,
    "activation": "tanh",
    "features": [
        "bruises",
        "odor",
        "cap-shape",
        "cap-color",
        "gill-spacing",
        "gill-size",
        "gill-color",
    ],
    "target": "class",
    "data": data.decode("utf-8"),
}


if __name__ == "__main__":
    r = requests.post(
        "http://127.0.0.1:4000/neural-network",
        json=ARGS,  # Send the data as a JSON object
        headers=headers,
    )

    if r.status_code != 200:
        raise Exception(
            f"bad response from neural network training api: {r.text}")

    # now send the results to the plot api
    nn_results = r.json()
    plot_args = nn_results
    r = requests.post(
        "http://127.0.0.1:5000/neural-network",
        json=plot_args,
        headers=headers,
    )
    if r.status_code != 200:
        raise Exception(f"bad response from ml-vis api: {r.text}")

    with open("plot.png", "wb") as f:
        f.write(r.content)
