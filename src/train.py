import argparse
import time

from data import load_data
from models import get_model
from evaluate import evaluate
from utils import save_metrics

def main(model_name):
    X_train, X_test, y_train, y_test = load_data("data/data.csv")

    model = get_model(model_name)

    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    metrics = evaluate(model, X_test, y_test)
    metrics["train_time_sec"] = round(train_time, 3)

    save_metrics(metrics, model_name)
    print(f"{model_name} metrics:", metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    main(args.model)
