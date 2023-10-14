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
    ],
    "target": "class",
    "data": data.decode("utf-8"),
}


if __name__ == "__main__":
    r = requests.post(
        "https://ml-from-scratch-v2.onrender.com/neural-network",
        json=ARGS,  # Send the data as a JSON object
        headers=headers,
    )

    print(r.status_code)
    print(r.text)

    # now send the results to the plot api
    nn_results = r.json()
    plot_args = nn_results
    r = requests.post(
        "https://ml-vis.onrender.com/neural-network",
        json=plot_args,
        headers=headers,
    )

    with open("plot.svg", "wb") as f:
        f.write(r.content)
