"""Pacientes adultos de atención primaria con y sin diabetes tipo 2, sintéticos, para el laboratorio de Nodum.

Una fila por paciente: edad, sexo, antropometría, tensión, analítica y hábitos, y si tiene diabetes tipo
2. Nadie existe: cada paciente sale de un generador con semilla.

De dónde salen las cifras, todas públicas:

- Prevalencia de diabetes tipo 2 en adultos en torno al 14 % y que crece con la edad, el IMC, la
  cintura, los antecedentes familiares y el sedentarismo (estudio di@bet.es).
- Umbrales de la ADA: glucosa en ayunas ≥ 126 mg/dl o HbA1c ≥ 6,5 % es diabetes; 100-125 mg/dl o
  5,7-6,4 % es prediabetes. Las analíticas de cada grupo caen sobre todo en su franja.
- Rangos habituales de colesterol, HDL, triglicéridos y tensión en adultos.

Como en una historia clínica real, falta algún dato: sale en torno a un 4 % sin perímetro de cintura y
un 7 % sin perfil lipídico (colesterol, HDL y triglicéridos a la vez).

Semilla fija y CSV con orden estable: los bytes salen idénticos en cada ejecución, así el checksum que
calcula el nodo no cambia. Lo publicado en `v1` no se toca nunca (SHA256SUMS lo vigila): un cambio en
los datos es un `v2`.

Uso: python3 generadores/diabetes_atencion_primaria.py <raíz_publicada>
"""

import csv
import io
import math
import random
import sys
from pathlib import Path

SEED = 20261024
PATIENTS = 2000
# Desplazan el riesgo para que salga en torno a un 14 % con diabetes y un 20 % con prediabetes.
DIABETES_SHIFT = 1.35
PREDIABETES_SHIFT = 1.8

FIELDS = (
    "paciente_id", "edad", "sexo", "imc", "perimetro_cintura_cm", "presion_sistolica_mmhg",
    "presion_diastolica_mmhg", "glucosa_ayunas_mg_dl", "hba1c_pct", "colesterol_total_mg_dl",
    "hdl_mg_dl", "trigliceridos_mg_dl", "antecedentes_familiares", "actividad_fisica", "tabaco",
    "diabetes_tipo2",
)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def patient(index: int, rng: random.Random) -> dict[str, str]:
    age = int(clamp(rng.gauss(54, 16), 18, 90))
    sex = rng.choice(("F", "M"))
    activity = rng.choices(("baja", "moderada", "alta"), weights=(40, 42, 18))[0]
    smoking = rng.choices(("nunca", "exfumador", "actual"), weights=(50, 28, 22))[0]
    family = rng.random() < 0.3
    bmi = clamp(rng.gauss(26.5 + 0.03 * (age - 50) + (1.2 if activity == "baja" else 0), 4.5), 17, 48)
    waist = clamp((88 if sex == "M" else 78) + 2.3 * (bmi - 25) + rng.gauss(0, 6), 60, 150)

    # Riesgo: edad, IMC, cintura, antecedentes y sedentarismo; calibrado para un 14 % en el conjunto.
    risk = (
        -7.3 + 0.055 * age + 0.11 * (bmi - 25) + 0.02 * (waist - 90) + (0.9 if family else 0)
        + (0.5 if activity == "baja" else -0.4 if activity == "alta" else 0) + (0.2 if smoking == "actual" else 0)
    )
    diabetic = rng.random() < 1 / (1 + math.exp(-(risk + DIABETES_SHIFT)))
    prediabetic = not diabetic and rng.random() < 1 / (1 + math.exp(-(risk + PREDIABETES_SHIFT)))

    if diabetic:
        glucose, hba1c = rng.gauss(150, 28), rng.gauss(7.3, 0.9)
    elif prediabetic:
        glucose, hba1c = rng.gauss(110, 7), rng.gauss(6.0, 0.22)
    else:
        glucose, hba1c = rng.gauss(89, 7), rng.gauss(5.3, 0.22)
    glucose, hba1c = clamp(glucose, 65, 320), clamp(hba1c, 4.2, 12.5)
    if diabetic and glucose < 126 and hba1c < 6.5:  # Diagnosticado: al menos uno de los dos lo dice.
        hba1c = round(rng.uniform(6.5, 7.2), 1)

    systolic = clamp(rng.gauss(118 + 0.45 * (age - 40) + 0.8 * (bmi - 25) + (6 if diabetic else 0), 13), 90, 200)
    diastolic = clamp(rng.gauss(74 + 0.35 * (bmi - 25), 8), 50, 120)
    cholesterol = clamp(rng.gauss(198 + 0.3 * (age - 50), 34), 110, 340)
    hdl = clamp(rng.gauss((48 if sex == "M" else 58) - 0.6 * (bmi - 25) - (5 if diabetic else 0), 11), 22, 110)
    triglycerides = clamp(rng.lognormvariate(math.log(110 + 4 * (bmi - 25) + (45 if diabetic else 0)), 0.38), 35, 900)

    no_waist = rng.random() < 0.04
    no_lipids = rng.random() < 0.06
    return {
        "paciente_id": f"P{index:05d}", "edad": str(age), "sexo": sex, "imc": f"{bmi:.1f}",
        "perimetro_cintura_cm": "" if no_waist else f"{waist:.0f}",
        "presion_sistolica_mmhg": f"{systolic:.0f}", "presion_diastolica_mmhg": f"{diastolic:.0f}",
        "glucosa_ayunas_mg_dl": f"{glucose:.0f}", "hba1c_pct": f"{hba1c:.1f}",
        "colesterol_total_mg_dl": "" if no_lipids else f"{cholesterol:.0f}",
        "hdl_mg_dl": "" if no_lipids else f"{hdl:.0f}",
        "trigliceridos_mg_dl": "" if no_lipids else f"{triglycerides:.0f}",
        "antecedentes_familiares": "si" if family else "no", "actividad_fisica": activity, "tabaco": smoking,
        "diabetes_tipo2": "1" if diabetic else "0",
    }


def main() -> None:
    out = Path(sys.argv[1]) / "sinteticos" / "diabetes-atencion-primaria" / "v1"
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(patient(i, rng) for i in range(1, PATIENTS + 1))
    (out / "pacientes.csv").write_text(buffer.getvalue(), encoding="utf-8")


if __name__ == "__main__":
    main()
