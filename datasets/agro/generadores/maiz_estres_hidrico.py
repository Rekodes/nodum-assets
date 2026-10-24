"""Fincas de maíz sintéticas con estrés hídrico, para el laboratorio de Nodum.

Tres fincas (una por nodo: nodeds1, nodeds2, nodeds3), una lectura por hora durante 30 días de
julio, en plena floración del maíz. Cada finca tiene su clima, su suelo y su forma de regar (datos
no IID, como en un FL real).

El agua del suelo sigue el balance de la FAO-56: el cultivo gasta Kc · ET0, la lluvia y el riego
reponen. `water_stress` es que el agotamiento supere el agua fácilmente disponible (p · TAW), como
lo mediría la sonda de humedad. La sonda no es una variable del modelo: lo que tiene que aprender
es que la cubierta de una planta con sed se calienta (cierra los estomas y no transpira), sobre
todo a mediodía y con el aire seco.

Semilla fija y JSON con claves ordenadas: los bytes salen idénticos en cada ejecución, así el
checksum que calcula el nodo no cambia. Lo publicado en `v1` no se toca nunca (SHA256SUMS lo
vigila): un cambio en los datos es un `v2`.

Uso: python3 generadores/maiz_estres_hidrico.py <raíz_publicada>
"""

import json
import math
import random
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

SEED = 20261020
START = datetime(2026, 7, 1, tzinfo=UTC)
HOURS = 30 * 24
KC_MID = 1.2  # Coeficiente del maíz a mitad de ciclo (FAO-56, tabla 12).
P_MAIZE = 0.55  # Fracción del agua útil que el maíz extrae sin estrés.
HEAT_WAVE_DAYS = range(11, 17)  # Del 12 al 17 de julio, +4 °C en las tres fincas.
FEATURES = ["air_temp_c", "relative_humidity_pct", "solar_radiation_w_m2", "wind_speed_m_s", "canopy_temp_c"]

FARMS = [
    # Pozo con cupo: riega poco y siempre igual, y el suelo arenoso guarda poca agua.
    {"farm_id": "finca-a", "node": "nodeds1", "name": "Llanura manchega, pozo con cupo",
     "lat": 39.05, "lon": -1.86, "temp_base": 25.5, "temp_amp": 8.0, "rh_base": 38.0,
     "wind_base": 2.6, "fc": 0.22, "wp": 0.09, "root_m": 0.9, "start_depletion": 30.0,
     "irrigation": {"method": "goteo", "every_days": 3, "hour": 6, "mm": 21.0}},
    # Riego a manta por turnos: el suelo se llena y se va vaciando hasta el turno siguiente.
    {"farm_id": "finca-b", "node": "nodeds2", "name": "Meseta del Duero, riego a manta",
     "lat": 41.62, "lon": -4.72, "temp_base": 22.0, "temp_amp": 8.5, "rh_base": 45.0,
     "wind_base": 3.0, "fc": 0.30, "wp": 0.15, "root_m": 0.9, "start_depletion": 15.0,
     "irrigation": {"method": "manta", "every_days": 11, "first_day": 4, "hour": 8, "mm": 85.0}},
    # Pivote que repone cada noche lo que falta, con un tope: en la ola de calor no llega.
    {"farm_id": "finca-c", "node": "nodeds3", "name": "Vega del Ebro, pivote",
     "lat": 41.71, "lon": -0.95, "temp_base": 24.0, "temp_amp": 6.5, "rh_base": 50.0,
     "wind_base": 3.4, "fc": 0.32, "wp": 0.16, "root_m": 0.9, "start_depletion": 45.0,
     "irrigation": {"method": "pivote", "every_days": 1, "hour": 22, "mm": 0.0, "max_mm": 6.8}},
]


def saturation_vapour_pressure_kpa(temp_c: float) -> float:
    return 0.6108 * math.exp(17.27 * temp_c / (temp_c + 237.3))


def et0_mm_h(temp_c: float, radiation_w_m2: float, wind_m_s: float, vpd_kpa: float) -> float:
    """ET0 horaria aproximada: Hargreaves con la radiación medida, corregida por viento y déficit de presión."""
    radiation_mm = radiation_w_m2 * 3600 / 1e6 / 2.45
    return max(0.0, 0.0135 * (temp_c + 17.8) * radiation_mm * (1 + 0.05 * (wind_m_s - 2) + 0.08 * (vpd_kpa - 2)))


def irrigation_mm(farm: dict, hour: int, depletion: float) -> float:
    plan = farm["irrigation"]
    day, hour_of_day = divmod(hour, 24)
    if hour_of_day != plan["hour"] or (day - plan.get("first_day", 0)) % plan["every_days"]:
        return 0.0
    if plan["method"] == "pivote":
        return round(min(depletion, plan["max_mm"]), 1)
    return plan["mm"]


def generate_farm(farm: dict, rng: random.Random) -> list[dict]:
    taw = 1000 * (farm["fc"] - farm["wp"]) * farm["root_m"]
    raw = P_MAIZE * taw
    depletion = farm["start_depletion"]
    wind = farm["wind_base"]
    cloud = 0.1
    storm_hours = {rng.randrange(14 * 24, 26 * 24) + h for h in range(3)}
    records = []
    for hour in range(HOURS):
        ts = START + timedelta(hours=hour)
        day, hour_of_day = divmod(hour, 24)
        heat = 4.0 if day in HEAT_WAVE_DAYS else 0.0
        # Temperatura: mínima al amanecer, máxima a las 16:00.
        temp = (farm["temp_base"] + heat + farm["temp_amp"] * math.sin(2 * math.pi * (hour_of_day - 10) / 24)
                + rng.gauss(0, 0.6))
        rh = min(98.0, max(8.0, farm["rh_base"] - 2.2 * (temp - farm["temp_base"]) + rng.gauss(0, 4)))
        wind = max(0.0, wind + 0.2 * (farm["wind_base"] - wind) + rng.gauss(0, 0.6))
        cloud = min(0.9, max(0.0, cloud + 0.1 * (0.1 - cloud) + rng.gauss(0, 0.05)))
        rain = round(rng.uniform(4, 12), 1) if hour in storm_hours else 0.0
        if rain:
            cloud = 0.85
        sun = max(0.0, math.sin(math.pi * (hour_of_day - 6) / 14)) if 6 <= hour_of_day <= 20 else 0.0
        radiation = 930 * sun * (1 - 0.75 * cloud)
        vpd = saturation_vapour_pressure_kpa(temp) * (1 - rh / 100)
        et0 = et0_mm_h(temp, radiation, wind, vpd)

        irrigation = irrigation_mm(farm, hour, depletion)
        # Ks de la FAO-56: con el agotamiento por encima del agua fácil, el cultivo transpira menos.
        ks = 1.0 if depletion <= raw else max(0.0, (taw - depletion) / (taw - raw))
        depletion = min(taw, max(0.0, depletion + KC_MID * ks * et0 - rain - irrigation))
        stressed = depletion > raw

        # Cubierta: con agua, más fresca que el aire si el aire está seco (línea base de Idso para el
        # maíz); con sed, cierra estomas y se calienta con el sol. Empieza a cerrarlos algo antes del
        # umbral y del todo al 75 % del agua útil: la señal avisa, pero no es la etiqueta.
        closure = min(1.0, max(0.0, (depletion / taw - 0.45) / 0.3))
        if sun > 0:
            canopy = temp + 3.11 - 1.97 * vpd + 8.0 * closure * radiation / 900 + rng.gauss(0, 0.8)
        else:
            canopy = temp - 1.0 + rng.gauss(0, 0.5)
        soil_moisture = 100 * (farm["fc"] - (depletion / taw) * (farm["fc"] - farm["wp"])) + rng.gauss(0, 0.4)

        records.append({
            "timestamp": ts.isoformat().replace("+00:00", "Z"),
            "farm_id": farm["farm_id"],
            "air_temp_c": round(temp, 2),
            "relative_humidity_pct": round(rh, 1),
            "solar_radiation_w_m2": round(radiation, 1),
            "wind_speed_m_s": round(wind, 2),
            "rain_mm": rain,
            "irrigation_mm": irrigation,
            "et0_mm": round(et0, 3),
            "canopy_temp_c": round(canopy, 2),
            "soil_moisture_pct": round(soil_moisture, 2),
            "water_stress": stressed,
        })
    return records


def describe(farm: dict) -> dict:
    taw = 1000 * (farm["fc"] - farm["wp"]) * farm["root_m"]
    return {
        **{k: farm[k] for k in ("farm_id", "node", "name", "lat", "lon")},
        "irrigation": farm["irrigation"]["method"],
        "soil": {"field_capacity": farm["fc"], "wilting_point": farm["wp"], "root_depth_m": farm["root_m"],
                 "taw_mm": round(taw, 1), "raw_mm": round(P_MAIZE * taw, 1)},
    }


def document(farms: list[dict], records: list[dict]) -> dict:
    return {
        "name": "Fincas de maíz sintéticas: estrés hídrico",
        "version": "1.0.0",
        "synthetic": True,
        "license": "CC-BY-4.0",
        "generator": {"script": "generadores/maiz_estres_hidrico.py", "seed": SEED},
        "period": {"start": START.isoformat().replace("+00:00", "Z"), "hours": HOURS, "step": "PT1H"},
        "crop": {"name": "maíz", "stage": "floración", "kc": KC_MID, "p": P_MAIZE},
        "target": {"field": "water_stress", "rule": "agotamiento del suelo > p · TAW (FAO-56)"},
        "features": FEATURES,
        "farms": [describe(f) for f in farms],
        "records": records,
    }


def write(path: Path, doc: dict) -> None:
    text = json.dumps(doc, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    path.write_text(text + "\n", encoding="utf-8")


def main() -> None:
    out = Path(sys.argv[1]) / "sinteticos" / "maiz-estres-hidrico" / "v1"
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    everything = []
    for farm in FARMS:
        everything.extend(generate_farm(farm, rng))
    write(out / "todas.json", document(FARMS, everything))


if __name__ == "__main__":
    main()
