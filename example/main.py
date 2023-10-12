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
    "activation": "sigmoid",
    "features": [
        "cap-shape",
        "cap-surface",
        "cap-color",
        "bruises",
        "odor",
        "gill-attachment",
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

    # now send the results to the plot api
    nn_results = r.json()
    plot_args = nn_results | ARGS
    r = requests.post(
        "http://127.0.0.1:3000/neural-network",
        json=plot_args,
        headers=headers,
    )

    print(r.content)

    with open("plot.svg", "wb") as f:
        f.write(r.content)
