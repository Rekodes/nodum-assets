"""Las dos lunas, repartidas en tres silos no-IID y un total para el brazo centralizado.

`total.csv` tiene exactamente las filas de los tres silos y `test.csv` es el test común, fuera de
todos. Desequilibrio de etiqueta: la clase 0 pesa 80 %, 50 % y 20 % en los silos a, b y c.

Semilla fija y CSV con orden estable: los bytes salen idénticos en cada ejecución. Lo publicado en
`v1` no se toca nunca (SHA256SUMS lo vigila): un cambio en los datos es un `v2`.

Uso: python3 generadores/lunas_federadas.py <raíz_publicada>
"""

import csv
import io
import math
import random
import sys
from pathlib import Path

SEED = 20261030
NOISE_STD = 0.2
ROWS_PER_SILO = 800
TEST_ROWS = 600
CLASS_0_SHARE = {"silo-a": 0.8, "silo-b": 0.5, "silo-c": 0.2}

type Row = tuple[str, str, int]


def moon_row(rng: random.Random, label: int) -> Row:
    angle = rng.uniform(0, math.pi)
    x1, x2 = (math.cos(angle), math.sin(angle)) if label == 0 else (1 - math.cos(angle), 0.5 - math.sin(angle))
    return f"{x1 + rng.gauss(0, NOISE_STD):.6f}", f"{x2 + rng.gauss(0, NOISE_STD):.6f}", label


def silo_rows(rng: random.Random, share: float) -> list[Row]:
    class_0 = round(ROWS_PER_SILO * share)
    rows = [moon_row(rng, 0) for _ in range(class_0)]
    rows += [moon_row(rng, 1) for _ in range(ROWS_PER_SILO - class_0)]
    rng.shuffle(rows)
    return rows


def write_csv(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(("x1", "x2", "label"))
    writer.writerows(rows)
    path.write_text(buffer.getvalue(), encoding="utf-8", newline="")


def main(root: Path) -> None:
    rng = random.Random(SEED)
    out = root / "sinteticos" / "lunas-federadas" / "v1"
    every: list[Row] = []
    for name, share in CLASS_0_SHARE.items():
        rows = silo_rows(rng, share)
        every += rows
        write_csv(out / f"{name}.csv", rows)
    rng.shuffle(every)
    write_csv(out / "total.csv", every)
    write_csv(out / "test.csv", [moon_row(rng, index % 2) for index in range(TEST_ROWS)])


if __name__ == "__main__":
    main(Path(sys.argv[1]))
