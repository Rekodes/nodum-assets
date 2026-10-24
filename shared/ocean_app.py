"""Lo común a todos los algoritmos: el contrato con ocean-node y la clase base.

Cada algoritmo hereda de `Algorithm`, declara sus parámetros e implementa `run`. `OceanApp` lo
ejecuta como espera el nodo: entradas en /data/inputs, parámetros en
/data/inputs/algoCustomData.json y salidas en /data/outputs.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal

log = logging.getLogger("ocean_app")

ParameterType = Literal["text", "number", "boolean", "select"]
Record = dict[str, object]


@dataclass(frozen=True)
class Parameter:
    """Un parámetro de consumo, como se declara al publicar el algoritmo en Nodum."""

    name: str
    type: ParameterType
    label: str
    description: str
    default: str | float | bool
    required: bool = False
    options: tuple[str, ...] = ()


class AlgorithmError(Exception):
    """Un fallo explicado en el log. La ejecución termina con `exit_code`."""

    def __init__(self, message: str, exit_code: int = 1) -> None:
        super().__init__(message)
        self.exit_code = exit_code


class InvalidParametersError(AlgorithmError):
    def __init__(self, message: str) -> None:
        super().__init__(message, exit_code=2)


class Parameters:
    """Los valores recibidos, leídos con el tipo y el valor por defecto que declara el algoritmo."""

    _TRUE = frozenset({"true", "1", "sí", "si", "yes"})

    def __init__(self, declared: tuple[Parameter, ...], received: dict[str, object]) -> None:
        self._declared = {p.name: p for p in declared}
        self.received = received

    def text(self, name: str) -> str:
        return str(self._value(name)).strip()

    def number(self, name: str) -> float:
        value = self._value(name)
        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return float(self._declared[name].default)

    def integer(self, name: str) -> int:
        return int(self.number(name))

    def flag(self, name: str) -> bool:
        value = self._value(name)
        if isinstance(value, str):
            return value.strip().lower() in self._TRUE
        return bool(value)

    def _value(self, name: str) -> object:
        value = self.received.get(name)
        return self._declared[name].default if value is None or value == "" else value


class Outputs:
    """La carpeta de salidas: lo único que sale del nodo."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def write_json(self, name: str, data: object) -> None:
        self.write_text(name, json.dumps(data, ensure_ascii=False, indent=2) + "\n")

    def write_text(self, name: str, text: str) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / name).write_text(text, encoding="utf-8")


@dataclass(frozen=True)
class Job:
    """Lo que recibe un algoritmo."""

    inputs: tuple[Path, ...]
    parameters: Parameters
    outputs: Outputs
    data_root: Path


class Algorithm(ABC):
    parameters: ClassVar[tuple[Parameter, ...]] = ()

    @abstractmethod
    def run(self, job: Job) -> None:
        """Lee `job.inputs`, escribe en `job.outputs` y deja en el log lo que ha hecho."""


class OceanApp:
    """Ejecuta un algoritmo como lo hace ocean-node."""

    PARAMETERS_FILE = "algoCustomData.json"

    def __init__(self, inputs: Path, outputs: Path) -> None:
        self._inputs = inputs
        self._outputs = outputs

    @classmethod
    def from_environment(cls) -> OceanApp:
        """Las carpetas del nodo, o las de OCEAN_INPUTS y OCEAN_OUTPUTS para probar en local."""
        return cls(
            Path(os.environ.get("OCEAN_INPUTS", "/data/inputs")),
            Path(os.environ.get("OCEAN_OUTPUTS", "/data/outputs")),
        )

    def run(self, algorithm: Algorithm) -> int:
        logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
        job = Job(
            inputs=self._input_files(),
            parameters=Parameters(algorithm.parameters, self._received_parameters()),
            outputs=Outputs(self._outputs),
            data_root=self._inputs.parent,
        )
        try:
            algorithm.run(job)
        except AlgorithmError as error:
            log.error("%s", error)
            return error.exit_code
        return 0

    def _input_files(self) -> tuple[Path, ...]:
        files = (p for p in self._inputs.rglob("*") if p.is_file() and p.name != self.PARAMETERS_FILE)
        return tuple(sorted(files))

    def _received_parameters(self) -> dict[str, object]:
        path = self._inputs / self.PARAMETERS_FILE
        try:
            data = json.loads(path.read_text(encoding="utf-8") or "{}")
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}


def read_records(path: Path) -> list[Record]:
    """Los registros de un JSON (lista de objetos o `{"records": [...]}`) o de un CSV con cabecera."""
    text = path.read_text(encoding="utf-8", errors="replace")
    if text.lstrip()[:1] not in ("[", "{"):
        return list(csv.DictReader(io.StringIO(text)))
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    rows = data.get("records", []) if isinstance(data, dict) else data
    return [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []


def main(algorithm: Algorithm) -> None:
    sys.exit(OceanApp.from_environment().run(algorithm))
