# nodum-assets

Los algoritmos genéricos y los datasets públicos del laboratorio de [Nodum](https://nodum.rekodes.com), el
dataspace de la tesis (compute-to-data con Ocean Protocol). Se publican, se confían y se ejecutan desde
Nodum (web o API `/v1`). Para publicarlos: [PUBLICAR.md](PUBLICAR.md), una ficha por asset.

## Organización

Primero, qué tipo de cosa es: `datasets/` (lo que se publica para que otros lo computen) y `algorithms/`
(lo que se ejecuta sobre ellos). Dentro, por temáticas.

| Carpeta | Qué hay |
|---|---|
| [`algorithms/generic/`](algorithms/generic/) | Los dos genéricos: valen para cualquier dataset y salen ya elegidos al publicar uno |
| [`algorithms/federated/`](algorithms/federated/) | FedAvg: el paso de entrenamiento que corre en cada sitio |
| [`datasets/agro/`](datasets/agro/) | Maíz: tres fincas con estrés hídrico, en un fichero |
| [`datasets/marine/`](datasets/marine/) | Pesca: un año de primera venta en lonjas de Galicia |
| [`datasets/health/`](datasets/health/) | Salud: pacientes de atención primaria con y sin diabetes tipo 2 |
| [`datasets/federated/`](datasets/federated/) | Lunas: tres silos no-IID, su unión para el centralizado y el test común |
| [`shared/`](shared/) | El contrato con ocean-node (`ocean_app.py`) y el Dockerfile de las imágenes |
| [`scripts/`](scripts/) | `push_images.py`, que sube las imágenes a Docker Hub |

## Algoritmos

Python, solo la biblioteca estándar y sin red. Dos formas:

- **Con imagen** (`line_count.py`): importa `shared/ocean_app.py`, que cumple el contrato con el nodo
  (entradas en `/data/inputs`, parámetros en `algoCustomData.json`, salidas en `/data/outputs`). Se
  construye con `shared/Dockerfile` como `docker.io/rekodes/nodum-<algoritmo>:0.2.0`.
- **Script** (`data_quality.py`, `fedavg_mlp.py`): un fichero suelto que corre en un entorno de Nodum (`python-3.15`).
  Se publica desde GitHub, con el commit fijado.

## Datasets

Sintéticos, con semilla fija y sin red: los bytes salen iguales en cada ejecución. Se sirven sin clave
desde `https://raw.githubusercontent.com/Rekodes/nodum-assets/main/`, con la ruta del repositorio. Lo publicado no cambia: un cambio es una versión nueva
(`v2`), y `SHA256SUMS` guarda la huella de cada fichero.

## Comandos

```bash
make data     # regenera los datasets y comprueba que no cambian
make lint     # ruff, mypy estricto en los de imagen y mypy en los scripts
make build    # construye las imágenes
make run a=line_count in="datasets/marine/sinteticos/primera-venta-galicia/v1/lonjas-2025.csv"
make push     # sube a tu Docker Hub las imágenes
```

## Licencia

Código: MIT ([`LICENSE`](LICENSE)). Datasets: CC BY 4.0 ([`LICENSE-DATA`](LICENSE-DATA)).
