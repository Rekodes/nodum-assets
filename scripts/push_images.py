"""Sube algoritmos a Docker Hub con tu cuenta y te dice qué poner en Nodum.

Cada uno sube a su espacio de Docker Hub (su usuario o una organización suya) con sus credenciales:
las de `docker login`, que guarda Docker, no este script. Por cada algoritmo:

1. Construye la imagen para linux/amd64, como la ejecuta el nodo.
2. La sube como `<espacio>/nodum-<algoritmo>:<versión>`.
3. Resuelve su digest sin credenciales, como lo hará Nodum: si no puede, el repositorio es privado y
   el nodo no podrá descargarla.
4. Imprime lo que pide Nodum al publicar un algoritmo de tipo imagen, con sus parámetros.

Una etiqueta que ya está en Docker Hub no se vuelve a subir (se salta y se dan sus datos): quien
confió en ella la tiene fijada por su digest. Para un cambio, una versión nueva (`--version`);
`--force` solo si nadie la usa todavía.

Uso (un algoritmo, una carpeta o todos; sin nada, pregunta):
  python3 scripts/push_images.py line_count
  python3 scripts/push_images.py generic --namespace mi-org --version 0.3.0
  python3 scripts/push_images.py all

Sin `docker login` hecho, lo pide. En CI o sin terminal: DOCKERHUB_USERNAME y DOCKERHUB_TOKEN (un
token de acceso de Docker Hub, no la contraseña).
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import inspect
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCKERFILE = ROOT / "shared" / "Dockerfile"
PLATFORM = "linux/amd64"
ENTRYPOINT = "python /app/main.py"
HUB_AUTH_KEY = "https://index.docker.io/v1/"
REGISTRY = "https://registry-1.docker.io"
MANIFEST_TYPES = ", ".join((
    "application/vnd.docker.distribution.manifest.v2+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.oci.image.index.v1+json",
))
NAMESPACE = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$")


class PushError(Exception):
    """Un fallo que se explica y termina el script."""


@dataclass(frozen=True)
class AlgorithmSource:
    path: Path

    @property
    def name(self) -> str:
        return self.path.stem

    @property
    def theme(self) -> str:
        return self.path.parent.name

    @property
    def build_arg(self) -> str:
        return str(self.path.relative_to(ROOT).with_suffix(""))

    def repository(self, namespace: str) -> str:
        return f"{namespace}/nodum-{self.name.replace('_', '-')}"

    def parameters(self) -> list[dict[str, object]]:
        """Los parámetros que declara el algoritmo, leídos de su clase (sin ejecutarlo)."""
        sys.path.insert(0, str(ROOT / "shared"))
        spec = importlib.util.spec_from_file_location(f"nodum_{self.name}", self.path)
        if spec is None or spec.loader is None:
            return []
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module  # dataclass busca su módulo aquí.
        spec.loader.exec_module(module)
        from ocean_app import Algorithm  # Solo se importa con shared/ en la ruta.

        for _, cls in inspect.getmembers(module, inspect.isclass):
            if issubclass(cls, Algorithm) and cls is not Algorithm and cls.__module__ == module.__name__:
                return [
                    {"name": p.name, "type": p.type, "label": p.label, "default": p.default,
                     "required": p.required, **({"options": list(p.options)} if p.options else {})}
                    for p in cls.parameters
                ]
        return []


def discover() -> list[AlgorithmSource]:
    """Los que llevan imagen: importan `ocean_app`. Un script corre en un entorno de Nodum, sin imagen."""
    paths = sorted(ROOT.glob("algorithms/*/*.py"))
    return [AlgorithmSource(p) for p in paths if "from ocean_app import" in p.read_text(encoding="utf-8")]


def run(command: list[str], *, quiet: bool = False, stdin: str | None = None) -> str:
    result = subprocess.run(
        command, cwd=ROOT, input=stdin, text=True,
        stdout=subprocess.PIPE if quiet else None, stderr=subprocess.PIPE if quiet else None, check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or "").strip().splitlines()[-1:] if quiet else []
        raise PushError(f"Falló «{' '.join(command[:3])} …»" + (f": {detail[0]}" if detail else ""))
    return result.stdout or ""


def check_docker() -> None:
    try:
        run(["docker", "info", "--format", "{{.ServerVersion}}"], quiet=True)
        run(["docker", "buildx", "version"], quiet=True)
    except FileNotFoundError as error:
        raise PushError("No encuentro docker. Instala Docker Desktop (o Docker Engine con buildx).") from error
    except PushError as error:
        raise PushError("Docker no responde: arráncalo y vuelve a probar.") from error


def logged_in_user() -> str | None:
    """El usuario de Docker Hub con el que está hecho `docker login`. Nunca lee la contraseña."""
    config_path = Path(os.environ.get("DOCKER_CONFIG", Path.home() / ".docker")) / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    helper = config.get("credHelpers", {}).get(HUB_AUTH_KEY) or config.get("credsStore")
    if helper:
        try:
            out = subprocess.run([f"docker-credential-{helper}", "get"], input=HUB_AUTH_KEY, text=True,
                                 capture_output=True, check=False)
        except FileNotFoundError:
            return None
        if out.returncode != 0:
            return None
        try:
            return str(json.loads(out.stdout).get("Username") or "") or None
        except json.JSONDecodeError:
            return None
    auth = config.get("auths", {}).get(HUB_AUTH_KEY, {}).get("auth")
    if not auth:
        return None
    return base64.b64decode(auth).decode("utf-8", errors="replace").split(":", 1)[0] or None


def ensure_login() -> str:
    user = logged_in_user()
    if user:
        return user
    username, token = os.environ.get("DOCKERHUB_USERNAME"), os.environ.get("DOCKERHUB_TOKEN")
    if username and token:
        run(["docker", "login", "-u", username, "--password-stdin"], quiet=True, stdin=token)
        return username
    if not sys.stdin.isatty():
        raise PushError(
            "No hay sesión en Docker Hub: haz `docker login` o define DOCKERHUB_USERNAME y DOCKERHUB_TOKEN."
        )
    print("No hay sesión en Docker Hub. Entra con tu cuenta (mejor con un token de acceso que con la contraseña).")
    run(["docker", "login"])
    user = logged_in_user()
    if not user:
        raise PushError("No he podido leer el usuario de Docker Hub tras el login.")
    return user


ALL = {"all", "todos"}


def select(available: list[AlgorithmSource], target: str) -> list[AlgorithmSource]:
    """Un algoritmo (`line_count`), una carpeta (`generic`) o todos (`all`)."""
    key = target.strip().lower().replace("-", "_")
    if key in ALL:
        return available
    in_theme = [a for a in available if a.theme == key]
    if in_theme:
        return in_theme
    named = [a for a in available if a.name == key]
    if named:
        return named
    themes = ", ".join(sorted({a.theme for a in available}))
    raise PushError(f"No conozco «{target}». Di un algoritmo, una carpeta ({themes}) o «all».")


def ask(available: list[AlgorithmSource]) -> list[AlgorithmSource]:
    themes = sorted({a.theme for a in available})
    for theme in themes:
        print(f"  {theme}/")
        for i, a in enumerate(available, 1):
            if a.theme == theme:
                print(f"    {i:>2}. {a.name}")
    answer = input(f"¿Qué subo? Un número, una carpeta ({', '.join(themes)}) o «todos»: ").strip()
    if answer.isdigit() and 1 <= int(answer) <= len(available):
        return [available[int(answer) - 1]]
    if not answer:
        raise PushError("No has elegido nada.")
    return select(available, answer)


def public_digest(repository: str, tag: str) -> str | None:
    """El digest como lo ve quien no tiene credenciales (Nodum y el nodo), o None si no lo ve.

    Con curl, que usa los certificados del sistema (el Python de python.org no los trae). Si Docker Hub
    no contesta, para: tomarlo por «no existe» haría volver a subir una etiqueta publicada.
    """
    token_url = f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repository}:pull"
    try:
        token = json.loads(curl("-fsS", token_url))["token"]
    except (PushError, KeyError, json.JSONDecodeError) as error:
        raise PushError("No puedo consultar Docker Hub; mira la conexión y vuelve a probar.") from error
    headers = curl("-sS", "-I", "-H", f"Accept: {MANIFEST_TYPES}", "-H", f"Authorization: Bearer {token}",
                   f"{REGISTRY}/v2/{repository}/manifests/{tag}")
    lines = headers.replace("\r", "").splitlines()
    status = lines[0].split()[1] if lines and len(lines[0].split()) > 1 else ""
    if status in {"401", "403", "404"}:
        return None
    if status != "200":
        raise PushError(f"Docker Hub contesta {status or 'nada'} por {repository}:{tag}; prueba más tarde.")
    for line in lines:
        name, _, value = line.partition(":")
        if name.lower() == "docker-content-digest":
            return value.strip()
    return None


def curl(*args: str) -> str:
    result = subprocess.run(["curl", "--max-time", "20", *args], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise PushError(f"curl falló: {result.stderr.strip()}")
    return result.stdout


def push(algorithm: AlgorithmSource, namespace: str, version: str, force: bool) -> dict[str, object]:
    repository = algorithm.repository(namespace)
    reference = f"{repository}:{version}"
    existing = public_digest(repository, version)
    if existing and not force:
        print(f"\n· {algorithm.name}: {reference} ya está en Docker Hub; no se vuelve a subir")
    else:
        print(f"\n→ {algorithm.name}: construyendo {reference} ({PLATFORM})")
        run(["docker", "buildx", "build", "--platform", PLATFORM, "-f", str(DOCKERFILE),
             "--build-arg", f"ALGORITHM={algorithm.build_arg}", "-t", reference, "--load", "."], quiet=True)
        print(f"→ {algorithm.name}: subiendo")
        run(["docker", "push", "-q", reference], quiet=True)
    digest = public_digest(repository, version)
    return {"algorithm": algorithm.name, "theme": algorithm.theme, "image": reference, "entrypoint": ENTRYPOINT,
            "digest": digest, "public": digest is not None, "pushed": not existing or force,
            "parameters": algorithm.parameters()}


def report(results: list[dict[str, object]]) -> None:
    print("\nPara publicarlos en Nodum (tipo «imagen»):")
    for r in results:
        state = "subida" if r["pushed"] else "ya estaba"
        print(f"\n  {r['algorithm']}  ({r['theme']} · {state})")
        print(f"    Imagen      {r['image']}")
        print(f"    Entrypoint  {r['entrypoint']}")
        if r["public"]:
            print(f"    Digest      {r['digest']}  (Nodum lo resuelve solo; sirve para comprobar)")
        else:
            repo = str(r["image"]).rsplit(":", 1)[0]
            print(f"    ⚠ No se ve sin credenciales: hazlo público en https://hub.docker.com/r/{repo} "
                  "(el nodo la descarga sin cuenta)")
        params: list[dict[str, object]] = r["parameters"]  # type: ignore[assignment]
        for p in params:
            extra = f" · opciones: {', '.join(map(str, p['options']))}" if "options" in p else ""  # type: ignore[call-overload]
            print(f"    Parámetro   {p['name']} ({p['type']}, por defecto {p['default']}){extra}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Sube algoritmos de nodum-assets a tu Docker Hub.")
    parser.add_argument("target", nargs="?",
                        help="Un algoritmo (line_count), una carpeta (generic) o «all». Sin nada, pregunta.")
    parser.add_argument("--namespace", help="Usuario u organización de Docker Hub. Por defecto, el del login.")
    parser.add_argument("--version", default="0.2.0", help="Etiqueta de las imágenes (por defecto 0.2.0).")
    parser.add_argument("--force", action="store_true", help="Vuelve a subir una etiqueta que ya existe.")
    parser.add_argument("--json", type=Path, help="Además, guarda el resultado en este fichero JSON.")
    args = parser.parse_args()

    try:
        check_docker()
        user = ensure_login()
        namespace = (args.namespace or user).lower()
        if not NAMESPACE.match(namespace):
            raise PushError(f"«{namespace}» no es un usuario u organización de Docker Hub válido.")
        available = discover()
        if args.target:
            selected = select(available, args.target)
        elif sys.stdin.isatty():
            selected = ask(available)
        else:
            raise PushError("Di qué subir: un algoritmo, una carpeta o «all».")
        print(f"Docker Hub: sesión de {user}; se sube a {namespace}/ con la etiqueta {args.version}")
        results = [push(a, namespace, args.version, args.force) for a in selected]
    except PushError as error:
        print(f"\n✗ {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    report(results)
    if args.json:
        args.json.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nGuardado en {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
