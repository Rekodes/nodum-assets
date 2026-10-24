# Mar: pesca

Un año de primera venta en lonjas de Galicia.

| Fichero | Filas | Qué es |
|---|---|---|
| `sinteticos/primera-venta-galicia/v1/lonjas-2025.csv` | 9986 | Una fila por lonja, especie, arte y día con venta en 2025 |

- **Lonjas:** A Coruña, Burela, Celeiro, Malpica, Muros, Ribeira, Cambados, Marín, Cangas y Vigo. Sin
  subasta en fin de semana ni el 1 y 6 de enero ni el 25 de diciembre.
- **Especies:** 14, con su nombre científico y su código FAO (lista ASFIS): merluza, pescada de pincho,
  rape, gallo, jurel, sardina, xouba, caballa, bonito del norte, pulpo, centolla, percebe, navaja y
  berberecho.
- **Campos:** `fecha`, `lonja`, `provincia`, `especie`, `nombre_cientifico`, `codigo_fao`, `arte`,
  `kilos`, `precio_medio_eur_kg`, `importe_eur`.
- **De dónde sale:** las épocas de cada especie (costera de la sardina, del bonito y de la caballa, vedas
  del pulpo, la navaja y el berberecho, marisco en invierno), qué arte entra en cada lonja y el orden de
  magnitud del precio siguen las estadísticas públicas de primera venta de la Xunta
  ([pescadegalicia.gal](https://www.pescadegalicia.gal)). Ninguna cifra es real: los kilos y los precios
  salen de un generador con semilla. El precio sube con poca oferta y el marisco se encarece en diciembre.

Generador: [`primera_venta_galicia.py`](generadores/primera_venta_galicia.py). URL:
`https://raw.githubusercontent.com/Rekodes/nodum-assets/main/datasets/marine/sinteticos/primera-venta-galicia/v1/lonjas-2025.csv`.
