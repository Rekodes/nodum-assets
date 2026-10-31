# Publicar en Nodum

Una ficha por asset, en el orden del formulario de **Publicar**. Lo que va en `código` se pega tal cual.

- **Primero los algoritmos:** los genéricos los publica un operador, con **Genérico** marcado. Así salen ya
  elegidos como de confianza al publicar los datasets.
- **Cabeceras de la petición:** ninguna; los datos se sirven sin clave.
- **Etiquetas:** la línea entera se pega de una vez; cada coma separa una etiqueta.

## Algoritmos

### Informe de calidad de datos

| Paso | Campo | Valor |
|---|---|---|
| Tipo | Qué publicas | Un algoritmo |
| | Tipo de algoritmo | Un script |
| Origen | Entorno | Python 3.15 |
| | De dónde sale | Un fichero de GitHub |
| | Fichero de GitHub | `github.com/Rekodes/nodum-assets/blob/main/algorithms/generic/data_quality.py` |
| Datos | Nombre | `Informe de calidad de datos` |
| | Descripción | Mide la calidad de cada fichero CSV, TSV, JSON o JSONL: celdas vacías, filas repetidas, tipos por columna, atípicos, fechas imposibles y filas que no cuadran con la cabecera. Da una nota de 0 a 100 con su desglose. No copia ningún valor de las celdas: solo recuentos, porcentajes y estadísticos numéricos. |
| | Etiquetas | `calidad-de-datos, benchmarking, perfilado, csv, json` |
| Nodos | Dónde podrá ejecutarse | Nodo 1, Nodo 2 y Nodo 3 |
| | Necesita internet al ejecutarse | No |
| | Formatos que lee | CSV, TSV, JSON y JSON Lines |
| | Genérico | Sí |
| Se prueba con | | Los dos datasets |

### Contar líneas

| Paso | Campo | Valor |
|---|---|---|
| Tipo | Qué publicas | Un algoritmo |
| | Tipo de algoritmo | Imagen y código de GitHub |
| Origen | Imagen con etiqueta | `docker.io/rekodes/nodum-line-count:0.2.0` |
| | Entrypoint | `python /app/main.py` |
| | Fichero de GitHub | `github.com/Rekodes/nodum-assets/blob/main/algorithms/generic/line_count.py` |
| Datos | Nombre | `Contar líneas` |
| | Descripción | Cuenta las líneas y los bytes de cada dataset de entrada y deja el resultado en line-count.json. |
| | Etiquetas | `generico, prueba` |
| Nodos | Dónde podrá ejecutarse | Nodo 1, Nodo 2 y Nodo 3 |
| | Necesita internet al ejecutarse | No |
| | Formatos que lee | CSV, TSV, JSON, JSON Lines, XML y Texto |
| | Genérico | Sí |
| Se prueba con | | Cualquiera |

**Parámetros**

| Nombre | Etiqueta | Tipo | Valor por defecto | Descripción | Obligatorio |
|---|---|---|---|---|---|
| `skip_header` | Saltar la cabecera | Sí o no | `No` | No cuenta la primera línea de cada fichero. | No |

### FedAvg · MLP

| Paso | Campo | Valor |
|---|---|---|
| Tipo | Qué publicas | Un algoritmo |
| | Tipo de algoritmo | Un script |
| Origen | Entorno | Python 3.15 |
| | De dónde sale | Un fichero de GitHub |
| | Fichero de GitHub | `github.com/Rekodes/nodum-assets/blob/main/algorithms/federated/fedavg_mlp.py` |
| Datos | Nombre | `FedAvg · MLP` |
| | Descripción | Un paso de FedAvg (Federated Averaging): entrena una MLP (Multilayer Perceptron, una red neuronal pequeña) con SGD (Stochastic Gradient Descent: ajusta los weights poco a poco, un batch de filas cada vez) a partir del modelo recibido y devuelve sus weights en update.json, sin ninguna fila. |
| | Etiquetas | `federated, fedavg, mlp, csv` |
| Nodos | Dónde podrá ejecutarse | Nodo 1, Nodo 2 y Nodo 3 |
| | Necesita internet al ejecutarse | No |
| | Formatos que lee | CSV |
| | Genérico | No |
| Parámetros | | Los de [su tabla](algorithms/federated/README.md) |

### FedAvg · MLP (PyTorch)

Lo mismo que FedAvg · MLP, entrenado con PyTorch sobre CPU. La imagen se construye sin emular y se sube a mano:
`make build-torch VERSION=0.3.0` y `docker push rekodes/nodum-fedavg-torch:0.3.0`.

| Paso | Campo | Valor |
|---|---|---|
| Tipo | Qué publicas | Un algoritmo |
| | Tipo de algoritmo | Solo una imagen |
| Origen | Imagen con etiqueta | `docker.io/rekodes/nodum-fedavg-torch:0.3.0` |
| | Entrypoint | `python /app/fedavg_torch.py` |
| Datos | Nombre | `FedAvg · MLP (PyTorch)` |
| | Descripción | Lo mismo que FedAvg · MLP, entrenado con PyTorch sobre CPU: mismos parámetros, salida y formato de weights, y prácticamente el mismo modelo. Un paso de FedAvg: entrena una MLP con SGD a partir del modelo recibido y devuelve sus weights en update.json, sin ninguna fila. Con mode=evaluate no entrena y solo devuelve métricas agregadas. |
| | Etiquetas | `federated, fedavg, mlp, csv, pytorch` |
| Nodos | Dónde podrá ejecutarse | Nodo 1, Nodo 2 y Nodo 3 |
| | Necesita internet al ejecutarse | No |
| | Formatos que lee | CSV |
| | Genérico | No |
| Parámetros | | Los de [su tabla](algorithms/federated/README.md) |

## Datasets

### Maíz · estrés hídrico en tres fincas

| Paso | Campo | Valor |
|---|---|---|
| Tipo | Qué publicas | Un dataset |
| Origen | Nodo | Nodo 1 |
| | URL del fichero | `https://raw.githubusercontent.com/Rekodes/nodum-assets/main/datasets/agro/sinteticos/maiz-estres-hidrico/v1/todas.json` |
| | Cabeceras de la petición | Ninguna |
| Datos | Nombre | `Maíz · estrés hídrico en tres fincas` |
| | Descripción | 2160 lecturas horarias de tres fincas de maíz en julio de 2026 (llanura manchega, meseta del Duero y vega del Ebro), cada una con su suelo y su riego. Ola de calor del 12 al 17 y una tormenta en cada finca. Sintético. |
| | Etiquetas | `maiz, finca, estres-hidrico, json` |
| Confianza | Algoritmos | Los dos genéricos (salen ya elegidos) |

### Pesca · primera venta en lonjas de Galicia (2025)

| Paso | Campo | Valor |
|---|---|---|
| Tipo | Qué publicas | Un dataset |
| Origen | Nodo | Nodo 1 |
| | URL del fichero | `https://raw.githubusercontent.com/Rekodes/nodum-assets/main/datasets/marine/sinteticos/primera-venta-galicia/v1/lonjas-2025.csv` |
| | Cabeceras de la petición | Ninguna |
| Datos | Nombre | `Pesca · primera venta en lonjas de Galicia (2025)` |
| | Descripción | 9986 ventas diarias de 14 especies en 10 lonjas gallegas durante 2025: kilos, precio medio por kilo e importe, con su arte y su código FAO. Épocas, vedas y precios aproximados a las estadísticas públicas de primera venta; ninguna cifra es real. Sintético. |
| | Etiquetas | `pesca, lonja, galicia, primera-venta, csv` |
| Confianza | Algoritmos | Los dos genéricos (salen ya elegidos) |

### Diabetes · pacientes de atención primaria

Lo publica otra cuenta, no la del operador: así se puede comprar su acceso desde la tuya.

| Paso | Campo | Valor |
|---|---|---|
| Tipo | Qué publicas | Un dataset |
| Origen | Nodo | Nodo 2 |
| | URL del fichero | `https://raw.githubusercontent.com/Rekodes/nodum-assets/main/datasets/health/sinteticos/diabetes-atencion-primaria/v1/pacientes.csv` |
| | Cabeceras de la petición | Ninguna |
| Datos | Nombre | `Diabetes · pacientes de atención primaria` |
| | Descripción | 2000 pacientes adultos sintéticos con edad, IMC, cintura, tensión, glucosa, HbA1c, perfil lipídico y hábitos, y si tienen diabetes tipo 2 (14 %). Prevalencia y umbrales de diagnóstico como en las cifras públicas (di@bet.es, ADA); nadie existe. Con huecos en cintura y lípidos, como una historia real. |
| | Etiquetas | `salud, diabetes, atencion-primaria, sintetico, csv` |
| Confianza | Algoritmos | Los dos genéricos (salen ya elegidos) |

### Lunas federadas · silo A

| Paso | Campo | Valor |
|---|---|---|
| Tipo | Qué publicas | Un dataset |
| Origen | Nodo | Nodo 1 |
| | URL del fichero | `https://raw.githubusercontent.com/Rekodes/nodum-assets/main/datasets/federated/sinteticos/lunas-federadas/v1/silo-a.csv` |
| | Cabeceras de la petición | Ninguna |
| Datos | Nombre | `Lunas federadas · silo A` |
| | Descripción | Dos clases en forma de luna, sintéticas. 800 filas con la clase 0 en el 80 %. |
| | Etiquetas | `federated, moons, synthetic, csv` |
| Confianza | Algoritmos | `FedAvg · MLP` |

### Lunas federadas · silo B

| Paso | Campo | Valor |
|---|---|---|
| Tipo | Qué publicas | Un dataset |
| Origen | Nodo | Nodo 2 |
| | URL del fichero | `https://raw.githubusercontent.com/Rekodes/nodum-assets/main/datasets/federated/sinteticos/lunas-federadas/v1/silo-b.csv` |
| | Cabeceras de la petición | Ninguna |
| Datos | Nombre | `Lunas federadas · silo B` |
| | Descripción | Dos clases en forma de luna, sintéticas. 800 filas con la clase 0 en el 50 %. |
| | Etiquetas | `federated, moons, synthetic, csv` |
| Confianza | Algoritmos | `FedAvg · MLP` |

### Lunas federadas · silo C

| Paso | Campo | Valor |
|---|---|---|
| Tipo | Qué publicas | Un dataset |
| Origen | Nodo | Nodo 3 |
| | URL del fichero | `https://raw.githubusercontent.com/Rekodes/nodum-assets/main/datasets/federated/sinteticos/lunas-federadas/v1/silo-c.csv` |
| | Cabeceras de la petición | Ninguna |
| Datos | Nombre | `Lunas federadas · silo C` |
| | Descripción | Dos clases en forma de luna, sintéticas. 800 filas con la clase 0 en el 20 %. |
| | Etiquetas | `federated, moons, synthetic, csv` |
| Confianza | Algoritmos | `FedAvg · MLP` |

### Lunas federadas · total

| Paso | Campo | Valor |
|---|---|---|
| Tipo | Qué publicas | Un dataset |
| Origen | Nodo | Nodo 1 |
| | URL del fichero | `https://raw.githubusercontent.com/Rekodes/nodum-assets/main/datasets/federated/sinteticos/lunas-federadas/v1/total.csv` |
| | Cabeceras de la petición | Ninguna |
| Datos | Nombre | `Lunas federadas · total` |
| | Descripción | Dos clases en forma de luna, sintéticas. 2400 filas, exactamente las de los tres silos, para el centralizado. |
| | Etiquetas | `federated, moons, synthetic, csv` |
| Confianza | Algoritmos | `FedAvg · MLP` |
