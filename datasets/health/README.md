# Salud: diabetes

Pacientes adultos de atención primaria con y sin diabetes tipo 2. Nadie existe: cada paciente sale de un
generador con semilla.

| Fichero | Filas | Qué es |
|---|---|---|
| `sinteticos/diabetes-atencion-primaria/v1/pacientes.csv` | 2000 | Un paciente por fila |

- **Campos:** `paciente_id`, `edad`, `sexo`, `imc`, `perimetro_cintura_cm`, `presion_sistolica_mmhg`,
  `presion_diastolica_mmhg`, `glucosa_ayunas_mg_dl`, `hba1c_pct`, `colesterol_total_mg_dl`, `hdl_mg_dl`,
  `trigliceridos_mg_dl`, `antecedentes_familiares`, `actividad_fisica`, `tabaco`, `diabetes_tipo2` (0 o 1).
- **Cifras:** 14,2 % con diabetes tipo 2 (5,9 % de 18 a 44 años, 12,1 % de 45 a 64 y 28,6 % desde 65) y
  21,9 % con prediabetes. Prevalencia y factores de riesgo como en el estudio di@bet.es (edad, IMC, cintura,
  antecedentes y sedentarismo).
- **Analítica:** umbrales de la ADA. Diabetes es glucosa en ayunas ≥ 126 mg/dl o HbA1c ≥ 6,5 %;
  prediabetes, 100-125 mg/dl o 5,7-6,4 %. Cada paciente diagnosticado cumple al menos uno de los dos.
- **Huecos, como en una historia real:** 90 pacientes sin perímetro de cintura y 150 sin perfil lipídico
  (colesterol, HDL y triglicéridos a la vez).

Generador: [`diabetes_atencion_primaria.py`](generadores/diabetes_atencion_primaria.py). URL:
`https://raw.githubusercontent.com/Rekodes/nodum-assets/main/datasets/health/sinteticos/diabetes-atencion-primaria/v1/pacientes.csv`.
