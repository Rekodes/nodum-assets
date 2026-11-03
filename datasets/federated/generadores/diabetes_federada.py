"""Diabetes tipo 2 en tres silos privados con poblaciones distintas, para probar el aprendizaje federado.

Parte del modelo de riesgo de `health/generadores/diabetes_atencion_primaria.py` (di@bet.es), sin
glucosa ni HbA1c: con ellas la etiqueta sale casi de un umbral. Lo que se predice es la diabetes a partir
de los factores de riesgo, y las clases están desequilibradas.

Los silos no son IID, y cada diferencia sale de la población, no de repartir al azar:

- Covariables: la edad media es 40, 55 y 70 años en los silos a, b y c, y el silo c es más sedentario.
- Etiqueta: por la edad, la prevalencia crece de un silo a otro.
- Tamaño: 1 500, 800 y 400 filas.
- Medición: el silo c mide la tensión sistólica unos 10 mmHg por encima y le faltan más datos.

`total.csv` tiene exactamente las filas de los tres silos. `test.csv` es el test común, fuera de todos,
con 200 filas de cada población. Semilla fija y CSV con orden estable: los bytes salen idénticos en cada
ejecución. Lo publicado en `v1` no se toca nunca (SHA256SUMS lo vigila): un cambio en los datos es un `v2`.

Uso: python3 generadores/diabetes_federada.py <raíz_publicada>
"""

import csv
import io
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

SEED = 20261103
DIABETES_SHIFT = 1.35
TEST_ROWS_PER_SILO = 200

FIELDS = (
    "edad", "sexo", "imc", "perimetro_cintura_cm", "presion_sistolica_mmhg", "presion_diastolica_mmhg",
    "colesterol_total_mg_dl", "hdl_mg_dl", "trigliceridos_mg_dl", "antecedentes_familiares", "actividad_fisica",
    "tabaco", "diabetes_tipo2",
)


@dataclass(frozen=True)
class Population:
    rows: int
    age_mean: float
    age_sd: float
    activity: tuple[int, int, int]  # Pesos de baja, moderada y alta.
    systolic_bias: float
    no_waist: float
    no_lipids: float


SILOS = {
    "silo-a": Population(1500, 40, 12, (25, 45, 30), 0, 0.02, 0.03),
    "silo-b": Population(800, 55, 13, (40, 42, 18), 0, 0.04, 0.06),
    "silo-c": Population(400, 70, 9, (60, 32, 8), 10, 0.10, 0.15),
}

type Row = dict[str, str]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def patient(rng: random.Random, population: Population) -> Row:
    age = int(clamp(rng.gauss(population.age_mean, population.age_sd), 18, 95))
    sex = rng.choice(("F", "M"))
    activity = rng.choices(("baja", "moderada", "alta"), weights=population.activity)[0]
    smoking = rng.choices(("nunca", "exfumador", "actual"), weights=(50, 28, 22))[0]
    family = rng.random() < 0.3
    bmi = clamp(rng.gauss(26.5 + 0.03 * (age - 50) + (1.2 if activity == "baja" else 0), 4.5), 17, 48)
    waist = clamp((88 if sex == "M" else 78) + 2.3 * (bmi - 25) + rng.gauss(0, 6), 60, 150)

    risk = (
        -7.3 + 0.055 * age + 0.11 * (bmi - 25) + 0.02 * (waist - 90) + (0.9 if family else 0)
        + (0.5 if activity == "baja" else -0.4 if activity == "alta" else 0) + (0.2 if smoking == "actual" else 0)
    )
    diabetic = rng.random() < 1 / (1 + math.exp(-(risk + DIABETES_SHIFT)))

    systolic = clamp(rng.gauss(118 + 0.45 * (age - 40) + 0.8 * (bmi - 25) + (6 if diabetic else 0), 13), 90, 200)
    systolic = clamp(systolic + population.systolic_bias, 90, 220)
    diastolic = clamp(rng.gauss(74 + 0.35 * (bmi - 25), 8), 50, 120)
    cholesterol = clamp(rng.gauss(198 + 0.3 * (age - 50), 34), 110, 340)
    hdl = clamp(rng.gauss((48 if sex == "M" else 58) - 0.6 * (bmi - 25) - (5 if diabetic else 0), 11), 22, 110)
    triglycerides = clamp(rng.lognormvariate(math.log(110 + 4 * (bmi - 25) + (45 if diabetic else 0)), 0.38), 35, 900)

    no_waist = rng.random() < population.no_waist
    no_lipids = rng.random() < population.no_lipids
    return {
        "edad": str(age), "sexo": sex, "imc": f"{bmi:.1f}",
        "perimetro_cintura_cm": "" if no_waist else f"{waist:.0f}",
        "presion_sistolica_mmhg": f"{systolic:.0f}", "presion_diastolica_mmhg": f"{diastolic:.0f}",
        "colesterol_total_mg_dl": "" if no_lipids else f"{cholesterol:.0f}",
        "hdl_mg_dl": "" if no_lipids else f"{hdl:.0f}",
        "trigliceridos_mg_dl": "" if no_lipids else f"{triglycerides:.0f}",
        "antecedentes_familiares": "si" if family else "no", "actividad_fisica": activity, "tabaco": smoking,
        "diabetes_tipo2": "1" if diabetic else "0",
    }


def write_csv(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(buffer.getvalue(), encoding="utf-8", newline="")


def main(root: Path) -> None:
    rng = random.Random(SEED)
    out = root / "sinteticos" / "diabetes-federada" / "v1"
    every: list[Row] = []
    test: list[Row] = []
    for name, population in SILOS.items():
        rows = [patient(rng, population) for _ in range(population.rows)]
        every += rows
        write_csv(out / f"{name}.csv", rows)
        test += [patient(rng, population) for _ in range(TEST_ROWS_PER_SILO)]
    rng.shuffle(every)
    rng.shuffle(test)
    write_csv(out / "total.csv", every)
    write_csv(out / "test.csv", test)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
