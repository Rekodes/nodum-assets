"""Cuenta las líneas y los bytes de cada dataset."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from ocean_app import Algorithm, AlgorithmError, Job, Parameter, main

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FileCount:
    file: str
    lines: int
    bytes: int

    @classmethod
    def of(cls, path: Path, skip_header: bool) -> FileCount:
        data = path.read_bytes()
        lines = data.count(b"\n") + (1 if data and not data.endswith(b"\n") else 0)
        if skip_header and lines:
            lines -= 1
        return cls(file=path.name, lines=lines, bytes=len(data))


class LineCount(Algorithm):
    parameters = (
        Parameter("skip_header", "boolean", "Saltar la cabecera", "No cuenta la primera línea de cada fichero.",
                  default=False),
    )

    def run(self, job: Job) -> None:
        if not job.inputs:
            raise AlgorithmError("No hay datasets de entrada")
        skip_header = job.parameters.flag("skip_header")
        counts = [FileCount.of(path, skip_header) for path in job.inputs]
        total = sum(c.lines for c in counts)

        job.outputs.write_json("line-count.json", {
            "datasets": len(counts),
            "total_lines": total,
            "skip_header": skip_header,
            "files": [asdict(c) for c in counts],
        })
        for c in counts:
            log.info("%s: %d líneas, %d bytes", c.file, c.lines, c.bytes)
        log.info("Total: %d líneas en %d datasets", total, len(counts))


if __name__ == "__main__":
    main(LineCount())
