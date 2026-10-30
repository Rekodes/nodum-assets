#!/usr/bin/env python3
# Un paso de FedAvg: entrena la MLP recibida y devuelve sus weights.
#
# MLP (multilayer perceptron): capas densas con ReLU entre ellas. Sin `schema`, 2-32-32-2: entran x1 y x2, salen los
# logits de las dos clases; 1 218 weights en total (ver LAYER_SIZES).
# Sirve para silo y centralizado: solo cambian el dataset y los epochs.
# SGD porque no tiene estado: Adam perdería sus momentos entre dos trabajos.
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
# La salida lleva pesos y métricas agregadas, nunca filas. Solo biblioteca estándar; no usa la red.

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

VERSION = "0.3.0"
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
type Layer = tuple[list[list[float]], list[float]]


def configure(new_schema: dict | None = None) -> None:
    """Fija el esquema y, con él, la arquitectura. Sin esquema, el de `DEFAULT_SCHEMA`."""
    global schema, LAYER_SIZES, N_WEIGHTS
    schema = new_schema or DEFAULT_SCHEMA
    n_inputs = len(schema["numeric"]) + sum(len(values) for values in schema["categorical"].values())
    if not n_inputs or len(schema["classes"]) < 2:
        raise ValueError("el esquema necesita al menos una columna de entrada y dos clases")
    LAYER_SIZES = (n_inputs, *schema["hidden"], len(schema["classes"]))
    N_WEIGHTS = sum((n_in + 1) * n_out for n_in, n_out in zip(LAYER_SIZES, LAYER_SIZES[1:], strict=False))


def init_weights(seed: int) -> list[float]:
    """Uniforme en ±1/√entradas, por capa, como las capas lineales de PyTorch."""
    rng = random.Random(seed)
    weights: list[float] = []
    for n_in, n_out in zip(LAYER_SIZES, LAYER_SIZES[1:], strict=False):
        bound = 1 / math.sqrt(n_in)
        weights += [rng.uniform(-bound, bound) for _ in range((n_in + 1) * n_out)]
    return weights


def encode_weights(weights: list[float]) -> str:
    return base64.b64encode(struct.pack(f"<{len(weights)}f", *weights)).decode("ascii")


def decode_weights(encoded: str) -> list[float]:
    raw = base64.b64decode(encoded, validate=True)
    if len(raw) != N_WEIGHTS * 4:
        raise ValueError(f"el modelo tiene {len(raw)} bytes y se esperaban {N_WEIGHTS * 4}")
    return list(struct.unpack(f"<{N_WEIGHTS}f", raw))


def _unflatten(weights: list[float]) -> list[Layer]:
    layers: list[Layer] = []
    position = 0
    for n_in, n_out in zip(LAYER_SIZES, LAYER_SIZES[1:], strict=False):
        matrix = [weights[position + row * n_in : position + (row + 1) * n_in] for row in range(n_out)]
        position += n_in * n_out
        layers.append((matrix, weights[position : position + n_out]))
        position += n_out
    return layers


def _flatten(layers: list[Layer]) -> list[float]:
    weights: list[float] = []
    for matrix, biases in layers:
        for row in matrix:
            weights += row
        weights += biases
    return weights


def _forward(layers: list[Layer], inputs: list[float]) -> list[list[float]]:
    """Las activaciones de cada capa; la última son los logits."""
    activations = [inputs]
    for index, (matrix, biases) in enumerate(layers):
        current = activations[-1]
        out = [sum(w * a for w, a in zip(row, current, strict=True)) + b for row, b in zip(matrix, biases, strict=True)]
        activations.append(out if index == len(layers) - 1 else [max(0.0, value) for value in out])
    return activations


def _probabilities(logits: list[float]) -> list[float]:
    top = max(logits)
    exps = [math.exp(value - top) for value in logits]
    total = sum(exps)
    return [value / total for value in exps]


def _add_gradient(layers: list[Layer], gradients: list[Layer], sample: Sample) -> None:
    inputs, label = sample
    activations = _forward(layers, inputs)
    probabilities = _probabilities(activations[-1])
    error = [p - (1.0 if index == label else 0.0) for index, p in enumerate(probabilities)]
    for index in range(len(layers) - 1, -1, -1):
        matrix, _ = layers[index]
        grad_matrix, grad_biases = gradients[index]
        below = activations[index]
        for row, signal in enumerate(error):
            grad_row = grad_matrix[row]
            for column, value in enumerate(below):
                grad_row[column] += signal * value
            grad_biases[row] += signal
        if index > 0:
            # Backprop: la ReLU de abajo solo propaga a sus neuronas activas.
            error = [
                sum(signal * matrix[row][column] for row, signal in enumerate(error)) if below[column] > 0 else 0.0
                for column in range(len(below))
            ]


def train(
    weights: list[float], samples: list[Sample], epochs: int, lr: float, batch_size: int, seed: int
) -> list[float]:
    """SGD por batches con cross-entropy."""
    layers = _unflatten(weights)
    rng = random.Random(seed)
    order = list(samples)
    for _ in range(epochs):
        rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            gradients = _unflatten([0.0] * len(weights))
            for sample in batch:
                _add_gradient(layers, gradients, sample)
            step = lr / len(batch)
            for (matrix, biases), (grad_matrix, grad_biases) in zip(layers, gradients, strict=True):
                for row, grad_row in zip(matrix, grad_matrix, strict=True):
                    for column, value in enumerate(grad_row):
                        row[column] -= step * value
                for row_index, value in enumerate(grad_biases):
                    biases[row_index] -= step * value
    return _flatten(layers)


def evaluate(weights: list[float], samples: list[Sample]) -> dict[str, float]:
    layers = _unflatten(weights)
    loss, hits = 0.0, 0
    for inputs, label in samples:
        logits = _forward(layers, inputs)[-1]
        loss -= math.log(max(_probabilities(logits)[label], 1e-12))
        hits += logits.index(max(logits)) == label
    return {"loss": loss / len(samples), "accuracy": hits / len(samples)}


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


def balanced_accuracy(weights: list[float], samples: list[Sample]) -> float:
    """Media de los aciertos de cada clase presente."""
    layers = _unflatten(weights)
    hits: dict[int, int] = {}
    totals: dict[int, int] = {}
    for inputs, label in samples:
        logits = _forward(layers, inputs)[-1]
        totals[label] = totals.get(label, 0) + 1
        hits[label] = hits.get(label, 0) + (logits.index(max(logits)) == label)
    return sum(hits[label] / totals[label] for label in totals) / len(totals)


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
