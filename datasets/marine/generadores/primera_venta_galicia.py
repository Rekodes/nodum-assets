"""Primera venta de pescado y marisco en lonjas de Galicia, sintética, para el laboratorio de Nodum.

Un año (2025) de subastas: una fila por lonja, especie, arte y día con venta. Las lonjas, las
especies con su código FAO (lista ASFIS), las artes, las épocas de cada especie (la costera de la
sardina en verano, la del bonito de julio a octubre, la centolla y el percebe en invierno, las
vedas del pulpo) y los órdenes de magnitud de precio por kilo siguen lo que publican las
estadísticas públicas de primera venta de la Xunta (pescadegalicia.gal). Ninguna cifra es real:
los kilos y los precios salen de un generador con semilla.

Las lonjas no subastan en fin de semana ni el 1 y el 6 de enero ni el 25 de diciembre. El precio
sube cuando hay poco y baja con mucha oferta, y en diciembre el marisco se encarece.

Semilla fija y CSV con orden estable: los bytes salen idénticos en cada ejecución, así el checksum
que calcula el nodo no cambia. Lo publicado en `v1` no se toca nunca (SHA256SUMS lo vigila): un
cambio en los datos es un `v2`.

Uso: python3 generadores/primera_venta_galicia.py <raíz_publicada>
"""

import csv
import io
import math
import random
import sys
from datetime import date, timedelta
from pathlib import Path

SEED = 20261024
YEAR = 2025
HOLIDAYS = ((1, 1), (1, 6), (12, 25))

# Lonja: provincia y lo que más se subasta allí (arrastre y palangre de altura, cerco o bajura).
PORTS = {
    "A Coruña": ("A Coruña", ("arrastre", "palangre", "cerco")),
    "Burela": ("Lugo", ("arrastre", "palangre", "cacea", "artes menores")),
    "Celeiro": ("Lugo", ("arrastre", "palangre", "cacea")),
    "Malpica": ("A Coruña", ("artes menores", "marisqueo")),
    "Muros": ("A Coruña", ("cerco", "artes menores", "marisqueo")),
    "Ribeira": ("A Coruña", ("cerco", "artes menores", "marisqueo")),
    "Cambados": ("Pontevedra", ("artes menores", "marisqueo")),
    "Marín": ("Pontevedra", ("arrastre", "palangre", "artes menores")),
    "Cangas": ("Pontevedra", ("cerco", "artes menores", "marisqueo")),
    "Vigo": ("Pontevedra", ("arrastre", "palangre", "cerco", "artes menores")),
}

# Especie: nombre científico, código FAO, arte, kilos medios en un día bueno de una lonja grande,
# precio medio en €/kg, meses de temporada (1-12) y meses de veda.
SPECIES = (
    ("Merluza", "Merluccius merluccius", "HKE", "arrastre", 2400, 6.5, range(1, 13), ()),
    ("Pescada de pincho", "Merluccius merluccius", "HKE", "palangre", 600, 11.0, range(1, 13), ()),
    ("Rape blanco", "Lophius piscatorius", "MON", "arrastre", 500, 9.5, range(1, 13), ()),
    ("Gallo", "Lepidorhombus whiffiagonis", "MEG", "arrastre", 450, 7.0, range(1, 13), ()),
    ("Jurel", "Trachurus trachurus", "HOM", "cerco", 3000, 1.6, range(1, 13), ()),
    ("Sardina", "Sardina pilchardus", "PIL", "cerco", 4000, 2.2, range(5, 11), ()),
    ("Xouba", "Sardina pilchardus", "PIL", "cerco", 1200, 2.8, range(4, 8), ()),
    ("Caballa", "Scomber scombrus", "MAC", "cerco", 5000, 0.9, range(2, 6), ()),
    ("Bonito del norte", "Thunnus alalunga", "ALB", "cacea", 2500, 5.5, range(7, 11), ()),
    ("Pulpo", "Octopus vulgaris", "OCC", "artes menores", 900, 8.0, range(1, 13), (5, 6)),
    ("Centolla", "Maja brachydactyla", "SCR", "artes menores", 150, 22.0, (1, 2, 3, 12), ()),
    ("Percebe", "Pollicipes pollicipes", "PCB", "marisqueo", 60, 45.0, (1, 2, 3, 4, 10, 11, 12), ()),
    ("Navaja", "Ensis magnus", "RAZ", "marisqueo", 120, 14.0, range(1, 13), (5, 6, 7)),
    ("Berberecho", "Cerastoderma edule", "COC", "marisqueo", 700, 5.0, range(1, 13), (4, 5, 6)),
)

# Lo que pesa cada lonja frente a la más grande.
PORT_SIZE = {
    "Vigo": 1.0, "A Coruña": 0.8, "Burela": 0.6, "Celeiro": 0.55, "Ribeira": 0.7, "Marín": 0.45,
    "Cangas": 0.3, "Muros": 0.3, "Malpica": 0.2, "Cambados": 0.25,
}

FIELDS = (
    "fecha", "lonja", "provincia", "especie", "nombre_cientifico", "codigo_fao", "arte",
    "kilos", "precio_medio_eur_kg", "importe_eur",
)


def days(year: int) -> list[date]:
    day, out = date(year, 1, 1), []
    while day.year == year:
        if day.weekday() < 5 and (day.month, day.day) not in HOLIDAYS:
            out.append(day)
        day += timedelta(days=1)
    return out


def season_factor(month: int, months: tuple[int, ...] | range) -> float:
    """Más en el centro de la temporada que en sus bordes."""
    months = tuple(months)
    if len(months) == 12:
        return 1.0
    middle = months[len(months) // 2]
    distance = min(abs(month - middle), 12 - abs(month - middle))
    return max(0.35, 1.0 - 0.18 * distance)


def rows(rng: random.Random) -> list[dict[str, str]]:
    out = []
    for day in days(YEAR):
        for port, (province, gears) in PORTS.items():
            for name, scientific, fao, gear, kilos, price, season, ban in SPECIES:
                if gear not in gears or day.month not in season or day.month in ban:
                    continue
                if rng.random() > 0.75:  # No todos los días entra cada especie.
                    continue
                supply = PORT_SIZE[port] * season_factor(day.month, season) * rng.lognormvariate(0, 0.45)
                kg = round(kilos * supply, 1)
                if kg < 1:
                    continue
                christmas = 1.35 if day.month == 12 and gear in ("artes menores", "marisqueo") else 1.0
                scarcity = 1.0 + 0.25 * math.tanh(1.0 - supply)
                eur_kg = round(price * christmas * scarcity * rng.uniform(0.85, 1.15), 2)
                out.append({
                    "fecha": day.isoformat(), "lonja": port, "provincia": province, "especie": name,
                    "nombre_cientifico": scientific, "codigo_fao": fao, "arte": gear,
                    "kilos": f"{kg:.1f}", "precio_medio_eur_kg": f"{eur_kg:.2f}",
                    "importe_eur": f"{kg * eur_kg:.2f}",
                })
    return out


def main() -> None:
    out = Path(sys.argv[1]) / "sinteticos" / "primera-venta-galicia" / "v1"
    out.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows(random.Random(SEED)))
    (out / "lonjas-2025.csv").write_text(buffer.getvalue(), encoding="utf-8")


if __name__ == "__main__":
    main()
