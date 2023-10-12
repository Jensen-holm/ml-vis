import requests

with open("mushrooms.csv", "rb") as csv:
    data = csv.read()

ARGS = {
    "epochs": 100,
    "hidden_size": 8,
    "learning_rate": 0.0001,
    "test_size": 0.1,
    "activation": "relu",
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
    )

    # now send the results to the plot api
    nn_results = r.json()
    r = requests.post(
        "http://127.0.0.1:3000/neural-network",
        json=nn_results,
    )

    # display results
    with open("plot.png", "wb") as f:
        f.write(r.content)
