# Algoritmos federados

Cada sitio corre el algoritmo sobre sus datos y quien orquesta agrega lo que devuelven.

| Algoritmo | Forma | Qué hace | Formatos | Salidas |
|---|---|---|---|---|
| [FedAvg · PyTorch](fedavg_torch.py) | Imagen `nodum-fedavg-torch` (`make build-torch`) | Lo mismo que FedAvg · MLP, con PyTorch sobre CPU: mismos parámetros, salida y formato de weights | CSV | `update.json` |
| [FedAvg · MLP](fedavg_mlp.py) | Script, entorno `python-3.15` | Entrena una MLP con SGD a partir del modelo recibido y devuelve sus weights. Sin `schema`, 2-32-32-2 (1 218 weights) | CSV | `update.json` |

| Parámetro | Tipo | Obligatorio | Qué hace |
|---|---|---|---|
| `model` | text | No | Los weights de entrada: float32 en little-endian y base64 (6 496 caracteres sin `schema`). Hace falta este o `model_file` |
| `model_file` | text | No | El nombre de un fichero entre las entradas que lleva el modelo, sin el tope de un parámetro |
| `mode` | text | No | `evaluate`: no entrena y solo devuelve `accuracy`, `loss`, `balanced_accuracy` y `n_eval` |
| `epochs` | number | Sí | Local epochs |
| `lr` | number | Sí | Tasa de aprendizaje |
| `batch_size` | number | Sí | Filas por batch |
| `seed` | number | Sí | Seed del barajado |
| `schema` | text | No | JSON con las columnas de entrada, igual en todos los sitios (ver el script): numéricas, categóricas y capas ocultas. Sin él, `x1,x2,label` |

`update.json` lleva los weights entrenados, `n_train`, la huella del modelo recibido y la loss y la accuracy
sobre el propio silo antes y después de entrenar. Nunca lleva filas.
