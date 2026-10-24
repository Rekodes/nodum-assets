#!/usr/bin/env python3
# Benchmarking de calidad de datos: algoritmo genérico para Nodum (Ocean Protocol).
#
# PRIVACIDAD (no negociable): corre sobre datos ajenos. La salida NUNCA incluye
# valores de celdas: ni ejemplos, ni mínimos/máximos de texto o fechas, ni
# valores frecuentes. Solo nombres de columna, conteos, porcentajes, tipos y
# estadísticos numéricos agregados (mín, máx, media, mediana, cuartiles) de
# columnas numéricas con al menos MIN_VALUES_FOR_STATS valores. Si la primera
# fila parece de datos, las columnas se nombran «columna_N». Los errores se
# registran por su clase, nunca por su mensaje (puede citar el dato).
#
# Entradas (ocean-node 4.5): /data/inputs/<i>/<fichero> y
# /data/inputs/algoCustomData.json con `nodum.datasets`. Salidas en /data/outputs.
# Solo biblioteca estándar; no usa la red.

import codecs
import csv
import io
import json
import math
import os
import random
import re
import sys
from array import array
from collections import Counter
from datetime import UTC, date, datetime

VERSION = "1.0.0"
DEFAULT_INPUTS = "/data/inputs"
# Solo para probar en local; en el nodo valen los valores por defecto.
INPUTS_DIR = os.environ.get("CALIDAD_INPUTS_DIR", DEFAULT_INPUTS)
OUTPUTS_DIR = os.environ.get("CALIDAD_OUTPUTS_DIR", "/data/outputs")

RESERVOIR_SIZE = 100_000  # valores por columna numérica para cuartiles y mediana
MIN_VALUES_FOR_STATS = 5  # por debajo no se publican estadísticos numéricos
MAX_ROW_HASHES = 5_000_000  # huellas de fila guardadas para detectar duplicados
KEY_HASH_BUDGET = 4_000_000  # huellas de valor repartidas entre columnas (claves)
MAX_CSV_COLUMNS = 1000
MAX_JSON_KEYS = 200  # más claves distintas pueden ser datos: ni se analizan ni se nombran
SAMPLE_BYTES = 64 * 1024
CHUNK = 1 << 20
JSON_OBJECT_MAX_BYTES = 64 << 20  # un JSON que no es una lista se lee entero
JSON_RECORD_MAX_BYTES = 64 << 20
MIN_DATE = date(1900, 1, 1)
TODAY = datetime.now(UTC).date()

WEIGHTS = {
    "completitud": 0.30,
    "unicidad": 0.15,
    "consistencia": 0.20,
    "validez": 0.15,
    "estructura": 0.20,
}
DIMENSIONS = {
    "completitud": "Celdas con valor sobre el total (filas × columnas).",
    "unicidad": "Filas que no repiten otra fila exacta.",
    "consistencia": "Valores no vacíos que cumplen el tipo dominante de su columna.",
    "validez": "Media de dos partes: valores numéricos y fechas sin atípicos (IQR) ni fechas "
    "fuera de 1900–hoy, y columnas que no son constantes.",
    "estructura": "Filas con el número de campos de la cabecera; −10 por cabecera vacía o "
    "repetida y −5 si la codificación no es UTF-8.",
}
NULL_MARKERS = frozenset({"", "na", "n/a", "nan", "null", "none", "nil", "-", "--", "?"})
BOOL_WORDS = frozenset({"true", "false", "verdadero", "falso", "sí", "si", "no", "yes"})
TYPES = ("entero", "decimal", "fecha", "booleano", "texto", "lista", "objeto")
SEPARATORS = {",": "coma", ";": "punto y coma", "\t": "tabulador", "|": "barra vertical"}

RE_INT = re.compile(r"[+-]?\d+")
RE_DEC = re.compile(r"[+-]?(?:\d+[.,]\d*|[.,]\d+|\d+)(?:[eE][+-]?\d+)?")
RE_INT_ES = re.compile(r"[+-]?\d{1,3}(?:\.\d{3}){2,}")  # 1.234.567
RE_DEC_ES = re.compile(r"[+-]?\d{1,3}(?:\.\d{3})+,\d+")  # 1.234,5
RE_DEC_EN = re.compile(r"[+-]?\d{1,3}(?:,\d{3})+\.\d+")  # 1,234.5
RE_ISO = re.compile(
    r"(\d{4})-(\d{2})-(\d{2})(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?"
)
RE_DMY = re.compile(r"(\d{1,2})([/-])(\d{1,2})\2(\d{4})(?: \d{1,2}:\d{2}(?::\d{2})?)?")


def log(msg):
    print(msg, flush=True)


# --- Clasificación de valores ---------------------------------------------


def _date(y, m, d):
    try:
        return date(int(y), int(m), int(d))
    except ValueError:
        return None


def classify_text(raw):
    """Devuelve (tipo, número, fecha); tipo None si la celda cuenta como vacía."""
    s = raw.strip()
    if len(s) <= 4 and s.lower() in NULL_MARKERS:
        return None, None, None
    c = s[0]
    if c.isdigit() or c in "+-.,":
        if RE_INT.fullmatch(s):
            num = float(s)
            return "entero", (num if math.isfinite(num) else None), None
        if RE_INT_ES.fullmatch(s):
            return "entero", float(s.replace(".", "")), None
        if RE_DEC_ES.fullmatch(s):
            return "decimal", float(s.replace(".", "").replace(",", ".")), None
        if RE_DEC_EN.fullmatch(s):
            return "decimal", float(s.replace(",", "")), None
        if RE_DEC.fullmatch(s):
            num = float(s.replace(",", "."))
            return "decimal", (num if math.isfinite(num) else None), None
        m = RE_ISO.fullmatch(s)
        if m:
            d = _date(m.group(1), m.group(2), m.group(3))
            if d:
                return "fecha", None, d
        m = RE_DMY.fullmatch(s)
        if m:
            d = _date(m.group(4), m.group(3), m.group(1))
            if d:
                return "fecha", None, d
        return "texto", None, None
    if s.lower() in BOOL_WORDS:
        return "booleano", None, None
    return "texto", None, None


def classify(v):
    if v is None:
        return None, None, None
    if isinstance(v, str):
        return classify_text(v)
    if isinstance(v, bool):
        return "booleano", None, None
    if isinstance(v, int):
        try:
            return "entero", float(v), None
        except OverflowError:
            return "entero", None, None
    if isinstance(v, float):
        return ("decimal", v, None) if math.isfinite(v) else (None, None, None)
    if isinstance(v, list):
        return ("lista" if v else None), None, None
    return ("objeto" if v else None), None, None


def value_hash(v):
    if isinstance(v, str):
        return hash(v.strip())
    if isinstance(v, (list, dict)):
        return hash(json.dumps(v, sort_keys=True, ensure_ascii=False, default=str))
    return hash((type(v).__name__, v))


# --- Perfil de un fichero (streaming) -------------------------------------


class Column:
    __slots__ = (
        "name",
        "seen",
        "empty",
        "types",
        "first_hash",
        "constant",
        "key_hashes",
        "key_state",
        "num_n",
        "num_sum",
        "num_min",
        "num_max",
        "reservoir",
        "dates",
        "dates_out",
    )

    def __init__(self, name):
        self.name = name
        self.seen = 0
        self.empty = 0
        self.types = Counter()
        self.first_hash = None
        self.constant = True
        self.key_hashes = set()
        self.key_state = "sí"
        self.num_n = 0
        self.num_sum = 0.0
        self.num_min = math.inf
        self.num_max = -math.inf
        self.reservoir = array("d")
        self.dates = 0
        self.dates_out = 0

    def observe(self, value, prof):
        self.seen += 1
        kind, num, day = classify(value)
        if kind is None:
            self.empty += 1
            return
        self.types[kind] += 1
        h = value_hash(value)
        if self.first_hash is None:
            self.first_hash = h
        elif self.constant and h != self.first_hash:
            self.constant = False
        keys = self.key_hashes
        if keys is not None:
            if h in keys:
                self.key_state, self.key_hashes = "no", None
            elif len(keys) >= prof.key_cap:
                self.key_state, self.key_hashes = "sin determinar", None
            else:
                keys.add(h)
        if num is not None:
            self.num_n += 1
            self.num_sum += num
            if num < self.num_min:
                self.num_min = num
            if num > self.num_max:
                self.num_max = num
            res = self.reservoir
            if len(res) < RESERVOIR_SIZE:
                res.append(num)
            else:
                j = prof.rng.randrange(self.num_n)
                if j < RESERVOIR_SIZE:
                    res[j] = num
        elif day is not None:
            self.dates += 1
            if day < MIN_DATE or day > TODAY:
                self.dates_out += 1


class Profile:
    def __init__(self):
        self.columns = {}
        self.rows = 0
        self.duplicates = 0
        self.row_hashes = set()
        self.row_hashes_full = False
        self.rng = random.Random(0)
        self.key_cap = KEY_HASH_BUDGET
        self.irregular = 0  # filas con otro número de campos / registros con otras claves
        self.bad_records = 0  # líneas o registros JSON que no son objetos válidos
        self.blank_lines = 0
        self.overflow_keys = set()
        self.notes = []

    def add_column(self, name):
        col = Column(name)
        self.columns[name] = col
        self.key_cap = max(100_000, KEY_HASH_BUDGET // len(self.columns))
        return col

    def add_row(self, h):
        self.rows += 1
        if h in self.row_hashes:
            self.duplicates += 1
        elif len(self.row_hashes) < MAX_ROW_HASHES:
            self.row_hashes.add(h)
        else:
            self.row_hashes_full = True


# --- Detección de codificación y formato ----------------------------------


def detect_encoding(path):
    """utf-8 (o utf-8-sig) si todo el fichero lo es; si no, latin-1 de respaldo."""
    dec = codecs.getincrementaldecoder("utf-8")()
    with open(path, "rb") as f:
        head = f.read(CHUNK)
        if head.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
            return "utf-16"
        if b"\x00" in head[:SAMPLE_BYTES]:
            return "binario"
        try:
            dec.decode(head)
            for chunk in iter(lambda: f.read(CHUNK), b""):
                dec.decode(chunk)
            dec.decode(b"", final=True)
        except UnicodeDecodeError:
            return "latin-1"
    return "utf-8-sig" if head.startswith(codecs.BOM_UTF8) else "utf-8"


def read_sample(path, encoding):
    with open(path, encoding=encoding, newline="") as f:
        sample = f.read(SAMPLE_BYTES)
        if f.read(1):
            cut = max(sample.rfind("\n"), sample.rfind("\r"))
            if cut > 0:
                sample = sample[:cut]
    return sample


def sniff_csv(sample):
    """Devuelve (dialecto, filas de muestra) o (None, motivo)."""
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        return None, "no se reconoce un separador de columnas"
    rows = [r for r in csv.reader(io.StringIO(sample), dialect) if r][:500]
    counts = Counter(len(r) for r in rows)
    if not counts:
        return None, "no hay filas"
    mode, freq = counts.most_common(1)[0]
    if mode < 2 or freq / len(rows) < 0.6:
        return None, "las líneas no tienen un número de campos regular (no parece una tabla)"
    cells = [c for r in rows for c in r if c.strip()]
    if cells and sum(len(c.split()) >= 4 for c in cells) * 2 > len(cells):
        return None, "parece texto libre: la mayoría de las celdas son frases"
    return dialect, rows


def header_is_data(first):
    """La primera fila es de datos si alguna celda es número o fecha."""
    return any(classify_text(c)[0] in ("entero", "decimal", "fecha") for c in first if c.strip())


# --- Lectores -------------------------------------------------------------


def profile_csv(path, encoding, dialect, sample_rows, info, prof):
    first = sample_rows[0]
    has_header = not header_is_data(first)
    ncols = len(first) if has_header else Counter(len(r) for r in sample_rows).most_common(1)[0][0]
    if ncols > MAX_CSV_COLUMNS:
        prof.notes.append(f"Tiene {ncols} columnas; se analizan las {MAX_CSV_COLUMNS} primeras.")
    ncols_used = min(ncols, MAX_CSV_COLUMNS)
    names, empty_headers, repeated = [], 0, []
    if has_header:
        seen = Counter()
        for i, raw in enumerate(first[:ncols_used], 1):
            name = " ".join(raw.split())[:60]
            if not name:
                empty_headers += 1
                name = f"(sin nombre {i})"
            seen[name] += 1
            if seen[name] == 2:
                repeated.append(name)
            names.append(name if seen[name] == 1 else f"{name} ({seen[name]})")
    else:
        names = [f"columna_{i}" for i in range(1, ncols_used + 1)]
    cols = [prof.add_column(n) for n in names]
    info.update(
        separador=SEPARATORS.get(dialect.delimiter, repr(dialect.delimiter)),
        cabecera=has_header,
        cabeceras_vacias=empty_headers,
        cabeceras_repetidas=repeated,
    )
    if not has_header:
        prof.notes.append(
            "La primera fila parece de datos: se analiza como fila y las columnas se llaman columna_N."
        )
    csv.field_size_limit(2**31 - 1)
    with open(path, encoding=encoding, newline="") as f:
        reader = csv.reader(f, dialect)
        if has_header:
            next(reader, None)
        try:
            for row in reader:
                if not row:
                    prof.blank_lines += 1
                    continue
                n = len(row)
                if n != ncols:
                    prof.irregular += 1
                for i, col in enumerate(cols):
                    col.observe(row[i] if i < n else "", prof)
                prof.add_row(hash(tuple(row)))
        except csv.Error as exc:
            prof.notes.append(
                f"La lectura se detuvo tras {prof.rows} filas: CSV mal formado ({type(exc).__name__})."
            )


def flatten(obj, prefix="", out=None, depth=0):
    out = {} if out is None else out
    for k, v in obj.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and v and depth < 3:
            flatten(v, key + ".", out, depth + 1)
        else:
            out[key] = v
    return out


def profile_records(records, prof, info):
    first_keys = None
    cols = prof.columns
    for rec in records:
        if rec is _BAD:
            prof.bad_records += 1
            continue
        if not isinstance(rec, dict):
            prof.bad_records += 1
            continue
        keys = frozenset(rec)
        if first_keys is None:
            first_keys = keys
        elif keys != first_keys:
            prof.irregular += 1
        prof.add_row(hash(json.dumps(rec, sort_keys=True, ensure_ascii=False, default=str)))
        flat = {" ".join(str(k).split())[:60]: v for k, v in flatten(rec).items()}
        for name, value in flat.items():
            col = cols.get(name)
            if col is None:
                if len(cols) >= MAX_JSON_KEYS:
                    if len(prof.overflow_keys) < 100_000:
                        prof.overflow_keys.add(hash(name))
                    continue
                col = prof.add_column(name)
                col.seen = prof.rows - 1  # filas anteriores sin la clave: vacías
                col.empty = prof.rows - 1
            col.observe(value, prof)
    # Filas en las que faltaba la clave: cuentan como celdas vacías.
    for col in cols.values():
        missing = prof.rows - col.seen
        col.seen += missing
        col.empty += missing
    if prof.overflow_keys:
        prof.notes.append(
            f"Hay más de {MAX_JSON_KEYS} claves distintas; {len(prof.overflow_keys)} no se "
            "analizan ni se nombran (pueden ser datos y no nombres de campo)."
        )


_BAD = object()


def iter_jsonl(f, prof):
    for line in f:
        line = line.strip()
        if not line:
            prof.blank_lines += 1
            continue
        try:
            yield json.loads(line)
        except ValueError:
            yield _BAD


def iter_json_array(f, prof):
    """Recorre una lista JSON de nivel superior sin cargarla entera."""
    dec = json.JSONDecoder()
    buf, pos, eof = f.read(CHUNK).lstrip("﻿"), 0, False
    pos = len(buf) - len(buf.lstrip())
    pos += 1  # '['
    while True:
        while True:
            while pos < len(buf) and buf[pos] in " \t\r\n,":
                pos += 1
            if pos < len(buf) or eof:
                break
            more = f.read(CHUNK)
            eof = not more
            buf, pos = buf[pos:] + more, 0
        if pos >= len(buf) or buf[pos] == "]":
            return
        try:
            obj, end = dec.raw_decode(buf, pos)
            if end < len(buf) or eof:
                yield obj
                pos = end
                if pos > CHUNK:
                    buf, pos = buf[pos:], 0
                continue
        except ValueError:
            if eof:
                prof.notes.append(f"JSON mal formado tras el registro {prof.rows}: se detiene la lectura.")
                prof.bad_records += 1
                return
        if len(buf) - pos > JSON_RECORD_MAX_BYTES:
            prof.notes.append(f"JSON mal formado o registro demasiado grande tras el registro {prof.rows}.")
            prof.bad_records += 1
            return
        more = f.read(CHUNK)
        eof = not more
        buf, pos = buf[pos:] + more, 0


def json_object_records(path, encoding, info):
    """JSON que es un objeto: toma su lista de objetos más larga."""
    if os.path.getsize(path) > JSON_OBJECT_MAX_BYTES:
        return None, "objeto JSON de más de 64 MB sin lista en la raíz: no se puede leer en streaming"
    with open(path, encoding=encoding) as f:
        try:
            data = json.load(f)
        except ValueError:
            return None, "JSON mal formado"
    if not isinstance(data, dict):
        return None, "JSON sin registros"
    best = None
    for key, value in data.items():
        tabular = isinstance(value, list) and value and sum(isinstance(x, dict) for x in value) * 2 > len(value)
        if tabular and (best is None or len(value) > len(data[best])):
            best = key
    if best is None:
        return None, "objeto JSON sin una lista de registros (no es tabular)"
    info["clave_registros"] = str(best)[:60]
    return data[best], None


# --- Resultados -----------------------------------------------------------


def quantile(vals, q):
    pos = (len(vals) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def finite(v):
    return v if v is not None and math.isfinite(v) else None


def pct(part, total):
    return round(100.0 * part / total, 2) if total else None


def dominant_type(types):
    nonempty = sum(types.values())
    if not nonempty:
        return None, 0
    numeric = types["entero"] + types["decimal"]
    best = max(TYPES, key=lambda t: types[t])
    if types["decimal"] and numeric >= types[best]:
        return "decimal", numeric
    return best, types[best]


def column_result(col, rows):
    nonempty = col.seen - col.empty
    kind, conforming = dominant_type(col.types)
    res = {
        "columna": col.name,
        "tipo": kind or "vacía",
        "vacias": col.empty,
        "vacias_pct": pct(col.empty, col.seen),
        "no_cumplen_tipo": nonempty - conforming,
        "no_cumplen_tipo_pct": pct(nonempty - conforming, nonempty),
        "tipos": {t: col.types[t] for t in TYPES if col.types[t]},
        "constante": bool(nonempty >= 2 and col.constant),
        "clave_candidata": None,
    }
    if col.empty == 0 and rows >= 2:
        res["clave_candidata"] = {"sí": True, "no": False}.get(col.key_state, "sin determinar")
    elif rows >= 2:
        res["clave_candidata"] = False
    if kind in ("entero", "decimal") and col.num_n >= MIN_VALUES_FOR_STATS:
        vals = sorted(col.reservoir)
        sampled = col.num_n > len(vals)
        q1, med, q3 = quantile(vals, 0.25), quantile(vals, 0.5), quantile(vals, 0.75)
        iqr = q3 - q1
        outliers = None
        if iqr > 0:
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            in_sample = sum(1 for v in vals if v < lo or v > hi)
            outliers = round(in_sample * col.num_n / len(vals))
        res["numerico"] = {
            "valores": col.num_n,
            "min": finite(col.num_min),
            "max": finite(col.num_max),
            "media": finite(col.num_sum / col.num_n),
            "mediana": med,
            "q1": q1,
            "q3": q3,
            "atipicos": outliers,
            "atipicos_pct": pct(outliers, col.num_n) if outliers is not None else None,
            "muestreado": sampled,
            "muestra": len(vals),
        }
    elif kind in ("entero", "decimal"):
        res["numerico"] = {
            "valores": col.num_n,
            "nota": f"menos de {MIN_VALUES_FOR_STATS} valores: sin estadísticos",
        }
    if kind == "fecha":
        res["fechas"] = {
            "valores": col.dates,
            "fuera_de_rango": col.dates_out,
            "fuera_de_rango_pct": pct(col.dates_out, col.dates),
        }
    return res


def scores(prof, columns, info):
    rows, ncols = prof.rows, len(columns)
    cells = rows * ncols
    empty = sum(c["vacias"] for c in columns)
    nonempty = cells - empty
    nonconf = sum(c["no_cumplen_tipo"] for c in columns)
    s = {}
    s["completitud"] = 100.0 * (1 - empty / cells) if cells else None
    s["unicidad"] = 100.0 * (1 - prof.duplicates / rows) if rows else None
    s["consistencia"] = 100.0 * (1 - nonconf / nonempty) if nonempty else None
    evaluated = anomalies = 0
    for c in columns:
        num = c.get("numerico")
        if num and num.get("atipicos") is not None:
            evaluated += num["valores"]
            anomalies += num["atipicos"]
        if c.get("fechas"):
            evaluated += c["fechas"]["valores"]
            anomalies += c["fechas"]["fuera_de_rango"]
    eligible = [c for c in columns if c["vacias"] <= rows - 2]
    parts = []
    if evaluated:
        parts.append(1 - anomalies / evaluated)
    if eligible:
        parts.append(1 - sum(c["constante"] for c in eligible) / len(eligible))
    s["validez"] = 100.0 * sum(parts) / len(parts) if parts else None
    base = rows + prof.bad_records
    if base:
        penalty = 10 * (info.get("cabeceras_vacias", 0) + len(info.get("cabeceras_repetidas", [])))
        penalty += 5 if info.get("codificacion", "").startswith("latin") else 0
        s["estructura"] = max(0.0, 100.0 * (1 - (prof.irregular + prof.bad_records) / base) - penalty)
    else:
        s["estructura"] = None
    used = {k: v for k, v in s.items() if v is not None}
    wsum = sum(WEIGHTS[k] for k in used)
    total = sum(WEIGHTS[k] * v for k, v in used.items()) / wsum if wsum else None
    breakdown = {
        k: {
            "puntuacion": round(v, 1) if v is not None else None,
            "peso": WEIGHTS[k],
            "peso_efectivo": round(WEIGHTS[k] / wsum, 4) if v is not None and wsum else 0,
            "aporte": round(WEIGHTS[k] / wsum * v, 1) if v is not None and wsum else 0,
        }
        for k, v in s.items()
    }
    return (round(total, 1) if total is not None else None), breakdown


def analyze_file(entry):
    path = entry["path"]
    info = {
        "fichero": os.path.basename(path),
        "dataset": entry.get("dataset"),
        "did": entry.get("did"),
        "posicion": entry.get("position"),
        "bytes": os.path.getsize(path),
    }
    if info["bytes"] == 0:
        return {**info, "estado": "no_analizado", "motivo": "fichero vacío"}
    enc = detect_encoding(path)
    if enc in ("binario", "utf-16"):
        motivo = "fichero binario (no es texto)" if enc == "binario" else "codificación UTF-16 no soportada"
        return {**info, "estado": "no_analizado", "motivo": motivo}
    info["codificacion"] = {
        "utf-8-sig": "utf-8 (con BOM)",
        "latin-1": "latin-1 (respaldo: no es UTF-8 válido)",
    }.get(enc, enc)
    sample = read_sample(path, enc)
    head = sample.lstrip("﻿ \t\r\n")
    if not head:
        return {**info, "estado": "no_analizado", "motivo": "solo contiene espacios"}
    prof = Profile()
    if head[0] == "[":
        info["formato"] = "JSON"
        with open(path, encoding=enc) as f:
            profile_records(iter_json_array(f, prof), prof, info)
    elif head[0] == "{":
        lines = [x for x in head.splitlines() if x.strip()]
        try:
            json.loads(lines[0])
            is_jsonl = len(lines) > 1  # una sola línea es un objeto JSON
        except ValueError:
            is_jsonl = False
        if is_jsonl:
            info["formato"] = "JSONL"
            with open(path, encoding=enc) as f:
                profile_records(iter_jsonl(f, prof), prof, info)
        else:
            info["formato"] = "JSON"
            records, motivo = json_object_records(path, enc, info)
            if motivo:
                return {**info, "estado": "no_analizado", "motivo": motivo}
            profile_records(iter(records), prof, info)
    elif head[0] == "<":
        return {**info, "estado": "no_analizado", "motivo": "parece XML o HTML: no es tabular"}
    else:
        dialect, rows = sniff_csv(sample)
        if dialect is None:
            return {**info, "estado": "no_analizado", "motivo": rows}
        info["formato"] = "TSV" if dialect.delimiter == "\t" else "CSV"
        profile_csv(path, enc, dialect, rows, info, prof)
    columns = [column_result(c, prof.rows) for c in prof.columns.values()]
    nota, breakdown = scores(prof, columns, info)
    keys = [c["columna"] for c in columns if c["clave_candidata"] is True]
    notes = list(prof.notes)
    if any(c.get("numerico", {}).get("muestreado") for c in columns):
        notes.append(
            f"Cuartiles, mediana y atípicos salen de una muestra aleatoria de {f_int(RESERVOIR_SIZE)} "
            "valores por columna (los atípicos se extrapolan); mín, máx y media son exactos."
        )
    if prof.row_hashes_full:
        notes.append(f"Duplicados: se comparan con las {f_int(MAX_ROW_HASHES)} primeras filas distintas.")
    if any(c["clave_candidata"] == "sin determinar" for c in columns):
        notes.append("Alguna columna tiene demasiados valores distintos para comprobar si es clave.")
    return {
        **info,
        "estado": "analizado",
        "filas": prof.rows,
        "columnas": len(columns),
        "nota": nota,
        "desglose": breakdown,
        "estructura": {
            "filas_irregulares": prof.irregular,
            "registros_invalidos": prof.bad_records,
            "lineas_en_blanco": prof.blank_lines,
            "cabeceras_vacias": info.pop("cabeceras_vacias", 0),
            "cabeceras_repetidas": info.pop("cabeceras_repetidas", []),
        },
        "unicidad": {
            "filas_duplicadas": prof.duplicates,
            "filas_duplicadas_pct": pct(prof.duplicates, prof.rows),
            "claves_candidatas": keys,
        },
        "detalle_columnas": columns,
        "notas": notes,
    }


# --- Entradas -------------------------------------------------------------


def _inside(path):
    root = os.path.realpath(INPUTS_DIR)
    return os.path.realpath(path).startswith(root + os.sep)


def _rebase(path):
    if path and INPUTS_DIR != DEFAULT_INPUTS and path.startswith(DEFAULT_INPUTS + "/"):
        return INPUTS_DIR + path[len(DEFAULT_INPUTS) :]
    return path


def list_files(directory, skip_custom=False):
    found = []
    for root, dirs, files in os.walk(directory):
        dirs[:] = sorted(
            (d for d in dirs if not d.startswith(".")),
            key=lambda d: (not d.isdigit(), int(d) if d.isdigit() else 0, d),
        )
        for name in sorted(files):
            if name.startswith(".") or (skip_custom and root == directory and name == "algoCustomData.json"):
                continue
            found.append(os.path.join(root, name))
    return found


def load_inputs():
    params = {}
    custom = os.path.join(INPUTS_DIR, "algoCustomData.json")
    if os.path.isfile(custom):
        try:
            with open(custom, encoding="utf-8") as f:
                params = json.load(f)
        except ValueError:
            log("algoCustomData.json no es JSON válido: se ignora")
    if not isinstance(params, dict):
        params = {}
    manifest = params.get("nodum.datasets")
    entries = []
    if isinstance(manifest, list) and manifest:
        for pos, d in enumerate(manifest):
            if not isinstance(d, dict):
                continue
            path = _rebase(d.get("path"))
            folder = _rebase(d.get("dir")) or os.path.join(INPUTS_DIR, str(pos))
            files = [path] if path and os.path.isfile(path) else list_files(folder)
            for p in files:
                if _inside(p):
                    entries.append(
                        {"path": p, "dataset": d.get("name"), "did": d.get("did"), "position": pos}
                    )
    else:
        for p in list_files(INPUTS_DIR, skip_custom=True):
            rel = os.path.relpath(p, INPUTS_DIR).split(os.sep)
            pos = int(rel[0]) if len(rel) > 1 and rel[0].isdigit() else None
            entries.append({"path": p, "position": pos})
    return entries


# --- Informe --------------------------------------------------------------


def f_int(n):
    return f"{n:,}".replace(",", ".") if abs(n) >= 10_000 else str(n)


def f_pct(x):
    return "—" if x is None else f"{x:.1f} %".replace(".", ",")


def f_num(v):
    if v is None:
        return "—"
    if float(v).is_integer() and abs(v) < 1e15:
        return f_int(int(v))
    return f"{v:.6g}".replace(".", ",")


def md_cell(text):
    return str(text).replace("|", "\\|").replace("`", "'").replace("\n", " ")


def render_md(report):
    out = ["# Informe de calidad de datos", ""]
    out.append(
        f"Generado el {report['generado'][:16].replace('T', ' ')} UTC con «Calidad de datos» "
        f"{VERSION}. Ficheros: {len(report['ficheros'])}."
    )
    out += [
        "",
        "> **Privacidad.** Este informe no contiene ningún valor de las celdas: solo nombres de "
        "columna, conteos, porcentajes, tipos y estadísticos numéricos agregados.",
        "",
    ]
    out += [
        "## Resumen",
        "",
        "| Fichero | Dataset | Formato | Filas | Columnas | Nota |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for r in report["ficheros"]:
        ok = r["estado"] == "analizado"
        out.append(
            f"| {md_cell(r['fichero'])} | {md_cell(r.get('dataset') or '—')} | {r.get('formato', '—')} | "
            f"{f_int(r['filas']) if ok else '—'} | {r['columnas'] if ok else '—'} | "
            f"{f_num(r['nota']) if ok else 'no analizado'} |"
        )
    out += [
        "",
        "## Cómo se calcula la nota",
        "",
        "Cada dimensión puntúa de 0 a 100 y la nota es su media ponderada. Si una dimensión no se "
        "puede medir (por ejemplo, sin valores numéricos ni fechas), su peso se reparte entre las demás.",
        "",
        "| Dimensión | Peso | Qué mide |",
        "| --- | ---: | --- |",
    ]
    for k, w in WEIGHTS.items():
        out.append(f"| {k.capitalize()} | {f_pct(w * 100)} | {DIMENSIONS[k]} |")
    out += [
        "",
        "- Cuentan como vacías las celdas sin texto y los marcadores NA, N/A, NaN, null, None, nil, -, -- y ?.",
        "- Tipos: entero, decimal (acepta coma decimal), fecha (AAAA-MM-DD y DD/MM/AAAA), booleano "
        "(true/false, sí/no, verdadero/falso), texto; en JSON también lista y objeto. Entero y decimal "
        "juntos cuentan como decimal.",
        "- Atípico: fuera de [Q1 − 1,5·IQR, Q3 + 1,5·IQR]. Cuartiles sobre una muestra de hasta "
        f"{f_int(RESERVOIR_SIZE)} valores por columna; con menos de {MIN_VALUES_FOR_STATS} valores no se "
        "dan estadísticos.",
        "- Clave candidata: sin vacías y sin valores repetidos. Duplicados: filas idénticas, comparadas "
        "por una huella de 64 bits.",
        "",
    ]
    for i, r in enumerate(report["ficheros"], 1):
        out += render_file(i, r)
    return "\n".join(out).rstrip() + "\n"


def render_file(i, r):
    title = md_cell(r["fichero"]) + (f" ({md_cell(r['dataset'])})" if r.get("dataset") else "")
    out = [f"## {i}. {title}", ""]
    if r["estado"] != "analizado":
        out += [f"**No analizado:** {r['motivo']}. Tamaño: {f_int(r['bytes'])} bytes.", ""]
        return out
    st = r["estructura"]
    meta = [f"Formato {r['formato']}"]
    if r.get("separador"):
        meta.append(f"separador {r['separador']}")
    meta.append(f"codificación {r['codificacion']}")
    meta.append(f"{f_int(r['filas'])} filas × {r['columnas']} columnas")
    if "cabecera" in r:
        meta.append("con cabecera" if r["cabecera"] else "sin cabecera")
    if r.get("clave_registros"):
        meta.append(f"registros en la clave «{md_cell(r['clave_registros'])}»")
    out += [", ".join(meta) + ".", ""]
    out += [
        f"**Nota: {f_num(r['nota'])} / 100**",
        "",
        "| Dimensión | Puntuación | Peso | Aporte |",
        "| --- | ---: | ---: | ---: |",
    ]
    for k, d in r["desglose"].items():
        if d["puntuacion"] is None:
            out.append(f"| {k.capitalize()} | no se mide | {f_pct(d['peso'] * 100)} | — |")
        else:
            weight = f_pct(d["peso_efectivo"] * 100)
            out.append(f"| {k.capitalize()} | {f_num(d['puntuacion'])} | {weight} | {f_num(d['aporte'])} |")
    u = r["unicidad"]
    facts = [
        f"Filas duplicadas: {f_int(u['filas_duplicadas'])} ({f_pct(u['filas_duplicadas_pct'])}).",
        "Claves candidatas: " + (", ".join(md_cell(k) for k in u["claves_candidatas"]) or "ninguna") + ".",
        f"Filas con otro número de campos o claves: {f_int(st['filas_irregulares'])}.",
    ]
    if st["registros_invalidos"]:
        facts.append(f"Registros o líneas JSON no válidos: {f_int(st['registros_invalidos'])}.")
    if st["lineas_en_blanco"]:
        facts.append(f"Líneas en blanco: {f_int(st['lineas_en_blanco'])}.")
    if st["cabeceras_vacias"]:
        facts.append(f"Cabeceras vacías: {st['cabeceras_vacias']}.")
    if st["cabeceras_repetidas"]:
        facts.append("Cabeceras repetidas: " + ", ".join(md_cell(n) for n in st["cabeceras_repetidas"]) + ".")
    const = [c["columna"] for c in r["detalle_columnas"] if c["constante"]]
    if const:
        facts.append("Columnas constantes: " + ", ".join(md_cell(n) for n in const) + ".")
    out += [""] + [f"- {x}" for x in facts + r["notas"]] + [""]
    out += [
        "| Columna | Tipo | Vacías | Fuera de tipo | Clave | Observaciones |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for c in r["detalle_columnas"]:
        obs = []
        if len(c["tipos"]) > 1:
            obs.append(", ".join(f"{t} {f_int(n)}" for t, n in c["tipos"].items()))
        if c["constante"]:
            obs.append("constante")
        n = c.get("numerico") or {}
        if n.get("atipicos"):
            obs.append(f"{f_int(n['atipicos'])} atípicos")
        if c.get("fechas", {}).get("fuera_de_rango"):
            obs.append(f"{f_int(c['fechas']['fuera_de_rango'])} fechas fuera de rango")
        key = {True: "sí", False: "no", None: "—"}.get(c["clave_candidata"], c["clave_candidata"])
        out.append(
            f"| {md_cell(c['columna'])} | {c['tipo']} | {f_pct(c['vacias_pct'])} | "
            f"{f_pct(c['no_cumplen_tipo_pct'])} | {key} | {'; '.join(obs) or '—'} |"
        )
    nums = [c for c in r["detalle_columnas"] if c.get("numerico", {}).get("mediana") is not None]
    if nums:
        out += [
            "",
            "| Columna numérica | Valores | Mín | Máx | Media | Mediana | Q1 | Q3 | Atípicos |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for c in nums:
            n = c["numerico"]
            atip = (
                "IQR nulo"
                if n["atipicos"] is None
                else f"{f_int(n['atipicos'])} ({f_pct(n['atipicos_pct'])})"
            )
            out.append(
                f"| {md_cell(c['columna'])} | {f_int(n['valores'])} | {f_num(n['min'])} | {f_num(n['max'])} | "
                f"{f_num(n['media'])} | {f_num(n['mediana'])} | {f_num(n['q1'])} | {f_num(n['q3'])} | {atip} |"
            )
    out.append("")
    return out


def main():
    entries = load_inputs()
    log(f"Calidad de datos {VERSION}: {len(entries)} fichero(s) de entrada")
    results = []
    for i, entry in enumerate(entries, 1):
        name = os.path.basename(entry["path"])
        log(f"[{i}/{len(entries)}] {name}: analizando")
        try:
            r = analyze_file(entry)
        except Exception as exc:  # un fichero no tumba el informe; solo la clase, nunca el mensaje
            r = {
                "fichero": name,
                "dataset": entry.get("dataset"),
                "did": entry.get("did"),
                "posicion": entry.get("position"),
                "bytes": os.path.getsize(entry["path"]),
                "estado": "no_analizado",
                "motivo": f"error inesperado ({type(exc).__name__})",
            }
        if r["estado"] == "analizado":
            log(f"[{i}/{len(entries)}] {name}: {r['formato']}, {r['filas']} filas, nota {r['nota']}")
        else:
            log(f"[{i}/{len(entries)}] {name}: no analizado ({r['motivo']})")
        results.append(r)
    report = {
        "algoritmo": {"nombre": "Calidad de datos", "version": VERSION},
        "generado": datetime.now(UTC).isoformat(timespec="seconds"),
        "privacidad": "Sin valores de celdas: solo nombres de columna, conteos, porcentajes, tipos y "
        "estadísticos numéricos agregados.",
        "metodo": {
            "pesos": WEIGHTS,
            "dimensiones": DIMENSIONS,
            "muestra_reservorio": RESERVOIR_SIZE,
            "minimo_valores_estadisticos": MIN_VALUES_FOR_STATS,
            "fechas_plausibles": [MIN_DATE.isoformat(), TODAY.isoformat()],
        },
        "ficheros": results,
    }
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    with open(os.path.join(OUTPUTS_DIR, "calidad.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False, default=str)
    with open(os.path.join(OUTPUTS_DIR, "calidad.md"), "w", encoding="utf-8") as f:
        f.write(render_md(report))
    log(f"Informe escrito en {OUTPUTS_DIR}: calidad.json y calidad.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
