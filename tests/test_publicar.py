import json

import pytest

from src import publicar as pub


@pytest.fixture
def salida(tmp_path):
    (tmp_path / "modelo.joblib").write_bytes(b"modelo")
    (tmp_path / "test_metrics.json").write_text(json.dumps(
        {"modelo": "svm", "umbral_verdura": 0.3, "accuracy": 0.9, "balanced_accuracy": 0.85}))
    (tmp_path / "resultados_cv.csv").write_text("modelo\nsvm\n")
    return tmp_path


@pytest.fixture
def github(monkeypatch):
    """Simula la API de GitHub y guarda las llamadas."""
    llamadas = []

    def pedir(metodo, url, token, datos=None, tipo="application/json", timeout=300):
        llamadas.append((metodo, url, datos))
        if metodo == "POST" and url.endswith("/releases"):
            return {"id": 7, "upload_url": "https://uploads.github.com/repos/x/y/releases/7/assets{?name,label}"}
        if metodo == "POST" and "fallar" in github.opciones and "uploads" in url:
            raise RuntimeError("falló la subida")
        if metodo == "PATCH":
            return {"html_url": "https://github.com/x/y/releases/tag/modelo-1"}
        return {}

    monkeypatch.setattr(pub, "_pedir", pedir)
    github.opciones = set()
    github.llamadas = llamadas
    return github


def test_publica_si_no_hay_modelo_previo(salida, github, monkeypatch):
    monkeypatch.setattr(pub, "metricas_publicadas", lambda token, repo: None)
    assert pub.publicar(salida, "t", "x/y") == "https://github.com/x/y/releases/tag/modelo-1"
    metodos = [(m, u.split("?name=")[-1] if "uploads" in u else u.rsplit("/", 1)[-1]) for m, u, _ in github.llamadas]
    assert metodos == [("POST", "releases"), ("POST", "modelo.joblib"), ("POST", "test_metrics.json"),
                       ("POST", "resultados_cv.csv"), ("PATCH", "7")]
    creado = github.llamadas[0][2]
    assert creado["draft"] is True and creado["tag_name"].startswith("modelo-")
    assert github.llamadas[-1][2] == {"draft": False, "make_latest": "true"}


def test_no_publica_si_no_mejora(salida, github, monkeypatch):
    monkeypatch.setattr(pub, "metricas_publicadas", lambda token, repo: {"balanced_accuracy": 0.9, "release": "m"})
    assert pub.publicar(salida, "t", "x/y") is None
    assert github.llamadas == []
    assert pub.publicar(salida, "t", "x/y", forzar=True)


def test_borra_el_borrador_si_falla_una_subida(salida, github, monkeypatch):
    monkeypatch.setattr(pub, "metricas_publicadas", lambda token, repo: None)
    github.opciones.add("fallar")
    with pytest.raises(RuntimeError, match="falló la subida"):
        pub.publicar(salida, "t", "x/y")
    assert github.llamadas[-1][0] == "DELETE"


def test_falta_el_modelo(tmp_path):
    with pytest.raises(FileNotFoundError):
        pub.publicar(tmp_path, "t", "x/y")
