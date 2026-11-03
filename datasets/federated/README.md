# Federado

Datos sintéticos para probar el aprendizaje federado. Nadie existe y cada fichero sale de un generador con
seed.

## Lunas federadas

Dos clases en forma de luna, con 800 filas por silo y desequilibrio de etiqueta.

| Fichero | Filas | Clase 0 | Para qué |
|---|---|---|---|
| `silo-a.csv` | 800 | 80 % | Un sitio del federado |
| `silo-b.csv` | 800 | 50 % | Un sitio del federado |
| `silo-c.csv` | 800 | 20 % | Un sitio del federado |
| `total.csv` | 2 400 | 50 % | El centralizado: exactamente las filas de los tres silos |
| `test.csv` | 600 | 50 % | El test común. No se publica: lo evalúa quien orquesta |

El algoritmo que los usa es [`fedavg_mlp.py`](../../algorithms/federated/fedavg_mlp.py).

## Diabetes federada

Diabetes tipo 2 en tres silos privados con poblaciones distintas. Se predice la diabetes a partir de los factores de
riesgo, sin glucosa ni HbA1c, con clases desequilibradas, columnas categóricas y huecos. Los silos no son IID por la
población, no por un reparto al azar: cambian la edad, la prevalencia, el tamaño y, en el silo c, la medición.

| Fichero | Filas | Diabetes | Edad media | Para qué |
|---|---|---|---|---|
| `silo-a.csv` | 1 500 | 5,9 % | 40 | Un sitio del federado: población joven y activa |
| `silo-b.csv` | 800 | 14,2 % | 55 | Un sitio del federado: la población general |
| `silo-c.csv` | 400 | 25,8 % | 70 | Un sitio del federado: población mayor y sedentaria; la sistólica sale unos 10 mmHg alta y faltan más datos |
| `total.csv` | 2 700 | 11,3 % | 49 | El centralizado: exactamente las filas de los tres silos |
| `test.csv` | 600 | 15,7 % | 55 | El test común, con 200 filas de cada población. No se publica: lo evalúa quien orquesta |

- **Campos:** `edad`, `sexo`, `imc`, `perimetro_cintura_cm`, `presion_sistolica_mmhg`, `presion_diastolica_mmhg`,
  `colesterol_total_mg_dl`, `hdl_mg_dl`, `trigliceridos_mg_dl`, `antecedentes_familiares`, `actividad_fisica`, `tabaco`
  y `diabetes_tipo2` (0 o 1).
- **Por qué sirve para comparar métodos:** con las clases desequilibradas, la accuracy engaña (decir siempre 0 da un
  89 % en el total), así que se mira la balanced accuracy. Y con silos tan distintos, FedAvg puede alejarse del
  centralizado, que es lo que FedProx o SCAFFOLD intentan corregir.

Generador: [`diabetes_federada.py`](generadores/diabetes_federada.py), a partir del modelo de riesgo de
[`diabetes_atencion_primaria.py`](../health/generadores/diabetes_atencion_primaria.py).
