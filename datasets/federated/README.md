# Federado

Datos sintéticos para probar el aprendizaje federado. Nadie existe y cada fichero sale de un generador con
semilla.

## Lunas federadas

Dos clases en forma de luna, con 800 filas por silo y desequilibrio de etiqueta.

| Fichero | Filas | Clase 0 | Para qué |
|---|---|---|---|
| `silo-a.csv` | 800 | 80 % | Un sitio del brazo federado |
| `silo-b.csv` | 800 | 50 % | Un sitio del brazo federado |
| `silo-c.csv` | 800 | 20 % | Un sitio del brazo federado |
| `total.csv` | 2 400 | 50 % | El brazo centralizado: exactamente las filas de los tres silos |
| `test.csv` | 600 | 50 % | El test común. No se publica: lo evalúa quien orquesta |

El algoritmo que los usa es [`fedavg_mlp.py`](../../algorithms/federated/fedavg_mlp.py).
