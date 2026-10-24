# Algoritmos genéricos

Valen para cualquier dataset. En Nodum los marca un operador, y salen ya elegidos como de confianza al
publicar un dataset en sus nodos.

| Algoritmo | Forma | Qué hace | Formatos | Salidas |
|---|---|---|---|---|
| [Contar líneas](line_count.py) | Imagen `rekodes/nodum-line-count:0.2.0` | Cuenta las líneas y los bytes de cada fichero de entrada | Cualquiera | `line-count.json` |
| [Informe de calidad de datos](data_quality.py) | Script, entorno `python-3.15` | Nota de 0 a 100 por fichero: completitud, unicidad, consistencia, validez y estructura, con su desglose | CSV, TSV, JSON, JSONL | `calidad.json`, `calidad.md` |

| Algoritmo | Parámetro | Tipo | Por defecto | Qué hace |
|---|---|---|---|---|
| Contar líneas | `skip_header` | boolean | `false` | No cuenta la primera línea de cada fichero |

El informe de calidad no copia ningún valor de las celdas: solo nombres de columna, recuentos,
porcentajes, tipos y estadísticos numéricos agregados.
