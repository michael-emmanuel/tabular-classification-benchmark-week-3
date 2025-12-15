import json
import os

def save_metrics(metrics, model_name):
    os.makedirs("results", exist_ok=True)
    path = "results/metrics.json"

    if os.path.exists(path):
        with open(path) as f:
            all_metrics = json.load(f)
    else:
        all_metrics = {}

    all_metrics[model_name] = metrics

    with open(path, "w") as f:
        json.dump(all_metrics, f, indent=2)
