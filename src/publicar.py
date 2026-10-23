"""Publica el modelo entrenado como Release de GitHub, sin pasos manuales.

Crea un Release nuevo con modelo.joblib, test_metrics.json y resultados_cv.csv. Como
src/modelo.py descarga siempre el modelo del Release más reciente, en cuanto termina la
publicación todos (notebook de predicción, demo, línea de comandos) usan el modelo nuevo.

Por defecto solo publica si el modelo nuevo supera al publicado en accuracy balanceada de test,
para que nadie reemplace por error un modelo mejor. Con --forzar publica de todas formas.

Necesita un token de GitHub con permiso de escritura en el repo (ver docs/detalles.md, "Publicar el modelo").
En Kaggle se guarda en Secrets con el nombre GITHUB_TOKEN; aquí se lee de la variable de entorno.

Uso:  GITHUB_TOKEN=... python -m src.publicar --salida /kaggle/working/outputs
"""
import argparse
import json
import os
import subprocess
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = "11909/Clasificador-fruta-verdura"
API = "https://api.github.com"
METRICA = "balanced_accuracy"
ARCHIVOS = ("modelo.joblib", "test_metrics.json", "resultados_cv.csv")


class NoEncontrado(RuntimeError):
    pass


def _pedir(metodo: str, url: str, token: str, datos=None, tipo="application/json", timeout=300):
    cuerpo = json.dumps(datos).encode() if isinstance(datos, dict) else datos
    peticion = urllib.request.Request(url, data=cuerpo, method=metodo, headers={
        **({"Authorization": f"Bearer {token}"} if token else {}),
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "clasificador-fruta-verdura",
        **({"Content-Type": tipo} if cuerpo is not None else {}),
    })
    try:
        with urllib.request.urlopen(peticion, timeout=timeout) as respuesta:
            contenido = respuesta.read()
    except urllib.error.HTTPError as error:
        detalle = error.read().decode(errors="replace")[:300]
        if error.code in (401, 403):
            raise PermissionError(f"GitHub rechazó el token ({error.code}): {detalle}. Revisa que no haya "
                                  "vencido y que tenga permiso de escritura (Contents) en el repo.") from error
        if error.code == 404:
            raise NoEncontrado(f"GitHub respondió 404 en {metodo} {url}: {detalle}") from error
        raise RuntimeError(f"GitHub respondió {error.code} en {metodo} {url}: {detalle}") from error
    return json.loads(contenido) if contenido else None


def metricas_publicadas(token: str, repo: str = REPO) -> dict | None:
    """test_metrics.json del Release más reciente, o None si no hay Release o no lo trae."""
    try:
        ultimo = _pedir("GET", f"{API}/repos/{repo}/releases/latest", token)
    except NoEncontrado:
        return None
    for archivo in ultimo.get("assets", []):
        if archivo["name"] == "test_metrics.json":
            peticion = urllib.request.Request(archivo["browser_download_url"],
                                              headers={"User-Agent": "clasificador-fruta-verdura"})
            with urllib.request.urlopen(peticion, timeout=60) as respuesta:
                datos = json.load(respuesta)
            return {**datos, "release": ultimo["tag_name"]}
    return None


def _commit_actual() -> str | None:
    r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                       cwd=Path(__file__).resolve().parent)
    return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None


def _descripcion(nuevas: dict, anteriores: dict | None) -> str:
    filas = [f"| {k} | {nuevas[k]:.4f} |" for k in ("accuracy", METRICA, "f1_macro", "f1_fruta", "f1_verdura")
             if isinstance(nuevas.get(k), (int, float))]
    texto = (f"Publicado automáticamente desde el notebook de entrenamiento.\n\n"
             f"Modelo: **{nuevas.get('modelo', '?')}**, umbral de verdura {nuevas.get('umbral_verdura', 0.5)}.\n\n"
             "| Métrica en test | Valor |\n|---|---|\n" + "\n".join(filas))
    if anteriores and isinstance(anteriores.get(METRICA), (int, float)):
        texto += f"\n\nRelease anterior ({anteriores['release']}): {METRICA} {anteriores[METRICA]:.4f}."
    return texto + f"\n\n```json\n{json.dumps(nuevas, indent=2)}\n```"


def publicar(salida: str | os.PathLike, token: str, repo: str = REPO, forzar: bool = False) -> str | None:
    """Publica los archivos de `salida` como Release nuevo. Devuelve su enlace, o None si no
    se publicó porque el modelo nuevo no supera al publicado."""
    salida = Path(salida)
    faltan = [a for a in ARCHIVOS if not (salida / a).exists()]
    if faltan:
        raise FileNotFoundError(f"Faltan {faltan} en {salida}. ¿Terminó bien el entrenamiento?")
    nuevas = json.loads((salida / "test_metrics.json").read_text(encoding="utf-8"))

    anteriores = metricas_publicadas(token, repo)
    if anteriores and not forzar and isinstance(anteriores.get(METRICA), (int, float)):
        if nuevas.get(METRICA, 0) <= anteriores[METRICA]:
            print(f"No se publica: {METRICA} {nuevas.get(METRICA, 0):.4f} no supera la de "
                  f"{anteriores['release']} ({anteriores[METRICA]:.4f}). Usa forzar=True para publicar igual.")
            return None

    etiqueta = "modelo-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    datos = {"tag_name": etiqueta, "name": f"Modelo {etiqueta[7:]} ({nuevas.get('modelo', '?')})",
             "body": _descripcion(nuevas, anteriores), "draft": True}
    if commit := _commit_actual():
        datos["target_commitish"] = commit
    # Se crea como borrador y se publica al final: así el Release "más reciente" nunca queda
    # sin modelo.joblib si una subida falla a medias.
    release = _pedir("POST", f"{API}/repos/{repo}/releases", token, datos)
    try:
        subida = release["upload_url"].split("{")[0]
        for nombre in ARCHIVOS:
            print(f"subiendo {nombre}...")
            _pedir("POST", f"{subida}?name={nombre}", token, (salida / nombre).read_bytes(),
                   tipo="application/octet-stream")
        release = _pedir("PATCH", f"{API}/repos/{repo}/releases/{release['id']}", token,
                         {"draft": False, "make_latest": "true"})
    except Exception:
        _pedir("DELETE", f"{API}/repos/{repo}/releases/{release['id']}", token)
        raise
    print(f"publicado: {release['html_url']}")
    return release["html_url"]


def main():
    parser = argparse.ArgumentParser(description="Publica el modelo entrenado como Release de GitHub.")
    parser.add_argument("--salida", default="/kaggle/working/outputs", help="carpeta con modelo.joblib")
    parser.add_argument("--repo", default=REPO)
    parser.add_argument("--forzar", action="store_true", help="publica aunque no supere al modelo publicado")
    args = parser.parse_args()
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise SystemExit("Falta la variable de entorno GITHUB_TOKEN.")
    publicar(args.salida, token, args.repo, args.forzar)


if __name__ == "__main__":
    main()
