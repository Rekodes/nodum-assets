# Algoritmos federados

Cada sitio corre el algoritmo sobre sus datos y quien orquesta agrega lo que devuelven.

| Algoritmo | Forma | Qué hace | Formatos | Salidas |
|---|---|---|---|---|
| [FedAvg · MLP](fedavg_mlp.py) | Script, entorno `python-3.15` | Entrena una MLP 2-32-32-2 (1 218 pesos) con SGD a partir del modelo recibido y devuelve sus pesos | CSV `x1,x2,label` | `update.json` |

| Parámetro | Tipo | Obligatorio | Qué hace |
|---|---|---|---|
| `model` | text | Sí | Los pesos de entrada: float32 en little-endian y base64 (6 496 caracteres) |
| `epochs` | number | Sí | Épocas locales |
| `lr` | number | Sí | Tasa de aprendizaje |
| `batch_size` | number | Sí | Filas por minilote |
| `seed` | number | Sí | Semilla del barajado |

`update.json` lleva los pesos entrenados, `n_train`, la huella del modelo recibido y la pérdida y la exactitud
sobre el propio silo antes y después de entrenar. Nunca lleva filas.
