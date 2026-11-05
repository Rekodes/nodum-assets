#!/usr/bin/env python3
# Un paso de FedAvg con PyTorch: entrena la MLP recibida y devuelve sus weights.
#
# El mismo algoritmo que fedavg_mlp.py, su versión sin dependencias: mismas entradas, salida, formato de weights,
# pesos iniciales y barajado. Con la misma seed, los dos dan los mismos weights.
#
# MLP (multilayer perceptron): capas densas con ReLU entre ellas, en una clase `MLP(nn.Module)`.
# SGD porque no tiene estado: Adam perdería sus momentos entre dos trabajos.
# Cada lote de entrenamiento pasa por cuatro fases (ver `train`): 1 · predecir (forward), 2 · medir (pérdida),
# 3 · culpar (backward, lo calcula PyTorch) y 4 · corregir (el paso del optimizador).
#
# Entradas (ocean-node 4.5): el CSV en /data/inputs/<i>/ y los parámetros en /data/inputs/algoCustomData.json.
# Salida: /data/outputs/update.json.
#
# El modelo llega en el parámetro `model` o, con `model_file`, en un fichero de ese nombre entre las entradas.
# Con `mode` = `evaluate` no entrena: mide el modelo sobre el dataset y solo devuelve métricas agregadas.
#
# `schema` (parámetro opcional, JSON) dice qué columnas entran y es igual en todos los sitios:
#   target, classes: la columna objetivo y sus valores; una fila sin un valor de `classes` se ignora.
#   numeric: {columna: [media, desviación]}; un valor vacío o que no es un número queda en la media.
#   categorical: {columna: [valores]}, en one-hot; un valor vacío o desconocido deja todo a cero.
#   hidden: el tamaño de cada capa oculta.
# La salida lleva pesos y métricas agregadas, nunca filas. Corre en una imagen con PyTorch sobre CPU; no usa la red.

import base64
import csv
import hashlib
import json
import math
import os
import random
import struct
import sys
import time
from pathlib import Path

import torch
from torch import nn

VERSION = "0.3.1"
INPUTS_DIR = Path(os.environ.get("FEDAVG_INPUTS_DIR", "/data/inputs"))
OUTPUTS_DIR = Path(os.environ.get("FEDAVG_OUTPUTS_DIR", "/data/outputs"))
DEFAULT_SCHEMA: dict = {
    "target": "label",
    "classes": ["0", "1"],
    "numeric": {"x1": [0.0, 1.0], "x2": [0.0, 1.0]},
    "categorical": {},
    "hidden": [32, 32],
}
schema = DEFAULT_SCHEMA
LAYER_SIZES = (2, 32, 32, 2)
N_WEIGHTS = sum((n_in + 1) * n_out for n_in, n_out in zip(LAYER_SIZES, LAYER_SIZES[1:], strict=False))

type Sample = tuple[list[float], int]


# --- Configuración: el esquema fija las columnas y la forma de la red -------------------------------------------------


def configure(new_schema: dict | None = None) -> None:
    """Fija el esquema y, con él, la arquitectura. Sin esquema, el de `DEFAULT_SCHEMA`."""
    global schema, LAYER_SIZES, N_WEIGHTS
    schema = new_schema or DEFAULT_SCHEMA
    n_inputs = len(schema["numeric"]) + sum(len(values) for values in schema["categorical"].values())
    if not n_inputs or len(schema["classes"]) < 2:
        raise ValueError("el esquema necesita al menos una columna de entrada y dos clases")
    LAYER_SIZES = (n_inputs, *schema["hidden"], len(schema["classes"]))
    N_WEIGHTS = sum((n_in + 1) * n_out for n_in, n_out in zip(LAYER_SIZES, LAYER_SIZES[1:], strict=False))


# --- Los datos: de fila a números -------------------------------------------------------------------------------------


def _scaled(text: str, mean: float, spread: float) -> float:
    """Normalizado; 0, la media, si no es un número."""
    try:
        value = float(text)
    except ValueError:
        return 0.0
    return (value - mean) / (spread or 1.0) if math.isfinite(value) else 0.0


def _encode(row: dict[str, str]) -> list[float]:
    inputs = [_scaled(row[name], *stats) for name, stats in schema["numeric"].items()]
    for name, values in schema["categorical"].items():
        inputs += [1.0 if row[name] == value else 0.0 for value in values]
    return inputs


def read_samples(path: Path) -> list[Sample]:
    classes = {name: index for index, name in enumerate(schema["classes"])}
    with path.open(newline="", encoding="utf-8") as handle:
        return [(_encode(row), classes[row[schema["target"]]]) for row in csv.DictReader(handle)
                if row[schema["target"]] in classes]


# --- El modelo --------------------------------------------------------------------------------------------------------


class MLP(nn.Module):
    """Capas lineales con ReLU entre ellas; la última da los logits, una puntuación por clase."""

    def __init__(self, sizes: tuple[int, ...]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            nn.Linear(n_in, n_out, dtype=torch.float64) for n_in, n_out in zip(sizes, sizes[1:], strict=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        return self.layers[-1](x)


# --- Los pesos: crearlos, pasarlos al modelo y transportarlos ---------------------------------------------------------


def init_weights(seed: int) -> list[float]:
    """Uniforme en ±1/√entradas, por capa, como las capas lineales de PyTorch."""
    rng = random.Random(seed)
    weights: list[float] = []
    for n_in, n_out in zip(LAYER_SIZES, LAYER_SIZES[1:], strict=False):
        bound = 1 / math.sqrt(n_in)
        weights += [rng.uniform(-bound, bound) for _ in range((n_in + 1) * n_out)]
    return weights


def to_model(weights: list[float]) -> MLP:
    """Los weights en orden plano, por capa la matriz fila a fila y después el sesgo, dentro de un `MLP`."""
    model = MLP(LAYER_SIZES)
    position = 0
    with torch.no_grad():
        for layer in model.layers:
            n_out, n_in = layer.weight.shape
            matrix = torch.tensor(weights[position : position + n_in * n_out], dtype=torch.float64)
            layer.weight.copy_(matrix.reshape(n_out, n_in))
            position += n_in * n_out
            layer.bias.copy_(torch.tensor(weights[position : position + n_out], dtype=torch.float64))
            position += n_out
    return model


def from_model(model: MLP) -> list[float]:
    weights: list[float] = []
    for layer in model.layers:
        weights += layer.weight.detach().flatten().tolist() + layer.bias.detach().tolist()
    return weights


def encode_weights(weights: list[float]) -> str:
    return base64.b64encode(struct.pack(f"<{len(weights)}f", *weights)).decode("ascii")


def decode_weights(encoded: str) -> list[float]:
    raw = base64.b64decode(encoded, validate=True)
    if len(raw) != N_WEIGHTS * 4:
        raise ValueError(f"el modelo tiene {len(raw)} bytes y se esperaban {N_WEIGHTS * 4}")
    return list(struct.unpack(f"<{N_WEIGHTS}f", raw))


# --- El entrenamiento: 1 · predecir, 2 · medir, 3 · culpar, 4 · corregir ----------------------------------------------


def train(
    weights: list[float], samples: list[Sample], epochs: int, lr: float, batch_size: int, seed: int
) -> list[float]:
    """SGD por lotes con cross-entropy. Cada epoch baraja las filas con `seed`, como fedavg_mlp.py."""
    model = to_model(weights)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    inputs = torch.tensor([row for row, _ in samples], dtype=torch.float64)
    labels = torch.tensor([label for _, label in samples])
    rng = random.Random(seed)
    order = list(range(len(samples)))
    model.train()
    for _ in range(epochs):
        rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            logits = model(inputs[batch])  # 1 · PREDECIR (forward)
            loss = loss_fn(logits, labels[batch])  # 2 · MEDIR (pérdida)
            optimizer.zero_grad()
            loss.backward()  # 3 · CULPAR (backward): el gradiente de cada peso
            optimizer.step()  # 4 · CORREGIR (update): peso −= lr × gradiente
    return from_model(model)


# --- Medir el modelo --------------------------------------------------------------------------------------------------


def _predicted(weights: list[float], samples: list[Sample]) -> tuple[torch.Tensor, torch.Tensor]:
    """Las probabilidades de cada clase y la etiqueta de cada fila."""
    model = to_model(weights)
    model.eval()
    with torch.inference_mode():
        logits = model(torch.tensor([row for row, _ in samples], dtype=torch.float64))
    return torch.softmax(logits, dim=1), torch.tensor([label for _, label in samples])


def evaluate(weights: list[float], samples: list[Sample]) -> dict[str, float]:
    probabilities, labels = _predicted(weights, samples)
    right = probabilities[torch.arange(len(labels)), labels].clamp_min(1e-12)
    hits = probabilities.argmax(dim=1) == labels
    return {"loss": float(-right.log().mean()), "accuracy": float(hits.double().mean())}


def balanced_accuracy(weights: list[float], samples: list[Sample]) -> float:
    """Media de los aciertos de cada clase presente."""
    probabilities, labels = _predicted(weights, samples)
    hits = probabilities.argmax(dim=1) == labels
    present = labels.unique()
    return float(sum(hits[labels == label].double().mean() for label in present) / len(present))


# --- El trabajo en el nodo: entradas, entrenamiento y salida ----------------------------------------------------------


def _dataset_path(parameters: dict) -> Path:
    model_file = parameters.get("model_file")
    entries = parameters.get("nodum.datasets") or []
    manifest = [entry for entry in entries if Path(str(entry.get("path"))).name != model_file]
    if manifest and manifest[0].get("path"):
        return Path(manifest[0]["path"])
    found = sorted(path for path in INPUTS_DIR.glob("*/*") if path.is_file() and path.name != model_file)
    if not found:
        raise FileNotFoundError("no hay dataset de entrada")
    return found[0]


def _encoded_model(parameters: dict) -> str:
    model_file = parameters.get("model_file")
    if not model_file:
        return str(parameters["model"])
    found = [path for path in INPUTS_DIR.glob("*/*") if path.is_file() and path.name == Path(str(model_file)).name]
    if not found:
        raise FileNotFoundError("no ha llegado el fichero del modelo")
    return found[0].read_text(encoding="ascii").strip()


def _write_update(update: dict) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUTS_DIR / "update.json").write_text(json.dumps(update), encoding="utf-8")


def run() -> None:
    parameters = json.loads((INPUTS_DIR / "algoCustomData.json").read_text(encoding="utf-8"))
    given = parameters.get("schema")
    configure(json.loads(given) if given else None)
    encoded = _encoded_model(parameters)
    samples = read_samples(_dataset_path(parameters))
    weights = decode_weights(encoded)
    model_sha256 = hashlib.sha256(encoded.encode("ascii")).hexdigest()
    if parameters.get("mode") == "evaluate":
        _write_update({"version": VERSION, "mode": "evaluate", "n_eval": len(samples), "model_in_sha256": model_sha256,
                       **evaluate(weights, samples), "balanced_accuracy": balanced_accuracy(weights, samples)})
        print(f"n_eval={len(samples)}")
        return
    started = time.monotonic()
    trained = train(
        weights, samples, int(parameters["epochs"]), float(parameters["lr"]),
        int(parameters["batch_size"]), int(parameters["seed"]),
    )
    seconds = time.monotonic() - started
    update = {
        "version": VERSION,
        "n_train": len(samples),
        "model_in_sha256": model_sha256,
        "model": encode_weights(trained),
        "before": evaluate(weights, samples),
        "after": evaluate(trained, samples),
        "train_seconds": round(seconds, 3),
    }
    _write_update(update)
    print(f"n_train={update['n_train']} train_seconds={update['train_seconds']}")


if __name__ == "__main__":
    try:
        run()
    except Exception as error:
        # Solo la clase: el mensaje de un error de lectura puede citar el dato.
        print(f"fallo: {type(error).__name__}", file=sys.stderr)
        sys.exit(1)
