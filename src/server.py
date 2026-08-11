"""
server.py
---------
The FL server / aggregator. It NEVER sees any hospital's raw data --
it only receives model weight updates from each client every round and
combines them using FedAvg (weighted average by each client's dataset size).

Two ways to run this project:

1. SIMULATION (recommended for Colab / a single GPU, no real network needed):
   run this file directly -> `python src/server.py`
   Flower spins up all clients as virtual processes internally.

2. REAL DEPLOYMENT (multiple machines / processes, true separation):
   run `start_server_only()` on the server machine, then run
   `client.py --cid 0 ...`, `client.py --cid 1 ...` etc. on each hospital
   machine, pointing at the server's IP.
"""

import argparse
import os
import torch
import flwr as fl
from flwr.common import ndarrays_to_parameters

from model import build_model, get_model_parameters, set_model_parameters
from client import make_client_fn
from dataset import get_test_dataloader
from utils import evaluate


HISTORY = {"round": [], "accuracy": [], "loss": [], "f1": [], "auc": []}


def get_evaluate_fn(num_classes: int, test_data_dir: str):
    """Central evaluation: server tests the aggregated global model on a
    held-out test set after every round (this data is NOT any client's
    private training data -- just used to measure global model quality)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_loader, classes = get_test_dataloader(test_data_dir)

    def evaluate_fn(server_round, parameters, config):
        model = build_model(num_classes=num_classes, device=device)
        set_model_parameters(model, parameters)
        metrics = evaluate(model, test_loader, device)

        HISTORY["round"].append(server_round)
        HISTORY["accuracy"].append(metrics["accuracy"])
        HISTORY["loss"].append(metrics["loss"])
        HISTORY["f1"].append(metrics["f1"])
        HISTORY["auc"].append(metrics["auc"])

        print(f"[Round {server_round}] Global model -> "
              f"acc={metrics['accuracy']:.4f} f1={metrics['f1']:.4f} auc={metrics['auc']:.4f}")

        os.makedirs("saved_models", exist_ok=True)
        torch.save(model.state_dict(), f"saved_models/global_model_round{server_round}.pth")

        return metrics["loss"], metrics

    return evaluate_fn


def fit_config(server_round: int):
    """Config sent to every client before each round's local training."""
    return {"lr": 1e-4, "local_epochs": 1}


def build_strategy(num_classes: int, test_data_dir: str, num_clients: int,
                    fraction_fit: float = 1.0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    init_model = build_model(num_classes=num_classes, device=device)
    init_params = ndarrays_to_parameters(get_model_parameters(init_model))

    return fl.server.strategy.FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=0.5,
        min_fit_clients=max(1, int(num_clients * fraction_fit)),
        min_available_clients=num_clients,
        on_fit_config_fn=fit_config,
        evaluate_fn=get_evaluate_fn(num_classes, test_data_dir),
        initial_parameters=init_params,
    )


def run_simulation(data_dir: str, test_data_dir: str, num_clients: int = 5,
                    num_rounds: int = 10, num_classes: int = 2):
    """Everything (server + N virtual clients) runs in ONE process -- ideal
    for Colab where we only have a single GPU to simulate multiple hospitals."""
    client_fn = make_client_fn(data_dir, num_clients, num_classes)
    strategy = build_strategy(num_classes, test_data_dir, num_clients)

    client_resources = {"num_cpus": 1, "num_gpus": 1.0 / num_clients if torch.cuda.is_available() else 0}

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources=client_resources,
    )
    return HISTORY


def start_server_only(server_address: str, num_clients: int, num_rounds: int,
                       num_classes: int, test_data_dir: str):
    """Real multi-machine deployment: server waits for clients to connect."""
    strategy = build_strategy(num_classes, test_data_dir, num_clients)
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["simulation", "server_only"], default="simulation")
    parser.add_argument("--data_dir", type=str, default="data/train")
    parser.add_argument("--test_data_dir", type=str, default="data/test")
    parser.add_argument("--num_clients", type=int, default=5)
    parser.add_argument("--num_rounds", type=int, default=10)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--server_address", type=str, default="0.0.0.0:8080")
    args = parser.parse_args()

    if args.mode == "simulation":
        history = run_simulation(args.data_dir, args.test_data_dir,
                                  args.num_clients, args.num_rounds, args.num_classes)
        print("Training complete. History:", history)
    else:
        start_server_only(args.server_address, args.num_clients, args.num_rounds,
                           args.num_classes, args.test_data_dir)
