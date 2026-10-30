#!/usr/bin/env python3
# FedAvg con PyTorch: lo mismo que fedavg_mlp.py, del que usa todo menos el bucle de entrenamiento.
#
# Mismas entradas, salida, formato de weights y seeds de barajado: los dos scripts se pueden comparar.
# Corre en una imagen con PyTorch sobre CPU; no usa la red.

import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fedavg_mlp as base  # noqa: E402


def _network(weights: list[float]) -> nn.Sequential:
    layers: list[nn.Module] = []
    sizes = base.LAYER_SIZES
    for index, (n_in, n_out) in enumerate(zip(sizes, sizes[1:], strict=False)):
        layers.append(nn.Linear(n_in, n_out, dtype=torch.float64))
        if index < len(sizes) - 2:
            layers.append(nn.ReLU())
    network = nn.Sequential(*layers)
    for (matrix, biases), layer in zip(base._unflatten(weights), network[::2], strict=True):
        layer.weight.data = torch.tensor(matrix, dtype=torch.float64)
        layer.bias.data = torch.tensor(biases, dtype=torch.float64)
    return network


def train(weights: list[float], samples: list, epochs: int, lr: float, batch_size: int, seed: int) -> list[float]:
    """SGD por batches con cross-entropy, con el mismo barajado que el script sin PyTorch."""
    import random

    network = _network(weights)
    optimizer = torch.optim.SGD(network.parameters(), lr=lr)
    loss_of = nn.CrossEntropyLoss()
    rng = random.Random(seed)
    order = list(samples)
    for _ in range(epochs):
        rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            inputs = torch.tensor([row for row, _ in batch], dtype=torch.float64)
            labels = torch.tensor([label for _, label in batch])
            optimizer.zero_grad()
            loss_of(network(inputs), labels).backward()
            optimizer.step()
    layers = [(layer.weight.detach().tolist(), layer.bias.detach().tolist()) for layer in network[::2]]
    return base._flatten(layers)


if __name__ == "__main__":
    base.train = train
    try:
        base.run()
    except Exception as error:
        # Solo la clase: el mensaje de un error de lectura puede citar el dato.
        print(f"fallo: {type(error).__name__}", file=sys.stderr)
        sys.exit(1)
