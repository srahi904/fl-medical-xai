"""
client.py
---------
Defines a Flower `NumPyClient`. Each instance represents ONE simulated
hospital: it holds only its own local data slice and never shares raw
images — only model weight updates go to the server (this is the core
privacy-preserving property of Federated Learning).

Run standalone (real multi-process deployment) with:
    python src/client.py --cid 0 --data_dir data/train --num_clients 5

For simulation (all clients in one process, e.g. on Colab) see server.py's
`start_simulation` which calls this same client logic internally.
"""

import argparse
import torch
import flwr as fl

from model import build_model, get_model_parameters, set_model_parameters
from dataset import get_client_dataloaders
from utils import train_one_epoch, evaluate


class HospitalClient(fl.client.NumPyClient):
    """One FL client = one hospital's local model + local data."""

    def __init__(self, model, train_loader, val_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

    def get_parameters(self, config):
        return get_model_parameters(self.model)

    def fit(self, parameters, config):
        """Called by the server each communication round: local training step."""
        set_model_parameters(self.model, parameters)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=config.get("lr", 1e-4))

        local_epochs = config.get("local_epochs", 1)
        for _ in range(local_epochs):
            train_one_epoch(self.model, self.train_loader, optimizer, self.device)

        # Only weights are returned -- never the raw images.
        return get_model_parameters(self.model), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        set_model_parameters(self.model, parameters)
        metrics = evaluate(self.model, self.val_loader, self.device)
        return metrics["loss"], len(self.val_loader.dataset), metrics


def make_client_fn(data_dir: str, num_clients: int, num_classes: int, batch_size: int = 16):
    """
    Factory used by Flower simulation to spin up a client on demand given a
    client id (cid). Keeps dataset splitting logic centralized in one place.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    client_loaders, classes = get_client_dataloaders(
        data_dir, num_clients, batch_size=batch_size, non_iid=True
    )

    def client_fn(cid: str):
        cid = int(cid)
        model = build_model(num_classes=num_classes, device=device)
        loader = client_loaders[cid]
        # simple local split: 80% train / 20% val for this client's own data
        n_val = max(1, int(0.2 * len(loader.dataset)))
        n_train = len(loader.dataset) - n_val
        train_subset, val_subset = torch.utils.data.random_split(loader.dataset, [n_train, n_val])
        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        return HospitalClient(model, train_loader, val_loader, device).to_client()

    return client_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", type=int, required=True, help="Client id (which hospital)")
    parser.add_argument("--data_dir", type=str, default="data/train")
    parser.add_argument("--num_clients", type=int, default=5)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--server_address", type=str, default="127.0.0.1:8080")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaders, classes = get_client_dataloaders(args.data_dir, args.num_clients, non_iid=True)
    loader = loaders[args.cid]

    n_val = max(1, int(0.2 * len(loader.dataset)))
    n_train = len(loader.dataset) - n_val
    train_subset, val_subset = torch.utils.data.random_split(loader.dataset, [n_train, n_val])
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=16, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=16, shuffle=False)

    model = build_model(num_classes=args.num_classes, device=device)
    client = HospitalClient(model, train_loader, val_loader, device)

    fl.client.start_client(server_address=args.server_address, client=client.to_client())
