# Agro: maíz

Tres fincas de maíz, una por nodo, que no comparten sus datos.

| Fichero | Registros | Qué es |
|---|---|---|
| `sinteticos/maiz-estres-hidrico/v1/todas.json` | 2160 h | Las tres fincas, una lectura por hora en julio de 2026 |

- **Fincas:** llanura manchega (pozo con cupo, suelo arenoso, 26 % de horas con estrés), meseta del Duero
  (riego a manta cada 11 días, 13 %) y vega del Ebro (pivote cada noche con tope, 9 %).
- Ola de calor del 12 al 17 (+4 °C) y una tormenta en cada finca.
- **Campos:** `timestamp`, `farm_id`, `air_temp_c`, `relative_humidity_pct`, `solar_radiation_w_m2`,
  `wind_speed_m_s`, `rain_mm`, `irrigation_mm`, `et0_mm`, `canopy_temp_c`, `soil_moisture_pct`,
  `water_stress`.
- El agua del suelo sigue el balance de la FAO-56 (Kc 1,2; p 0,55). `water_stress` es que el agotamiento
  pase de p · TAW, lo que diría la sonda de humedad.
- Cada finca tiene su clima, su suelo y su riego: los datos no son IID.

Generador: [`maiz_estres_hidrico.py`](generadores/maiz_estres_hidrico.py). URL:
`https://raw.githubusercontent.com/Rekodes/nodum-assets/main/datasets/agro/sinteticos/maiz-estres-hidrico/v1/todas.json`.
