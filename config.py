"""Configuración central de WeatherML.

Todos los valores se pueden cambiar con variables de entorno
sin editar el código, por ejemplo:

    set WEATHERML_TEST_SIZE=0.25      (Windows)
    export WEATHERML_TEST_SIZE=0.25   (Linux / macOS)
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def _env(name, default, cast=str):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    if cast is bool:
        return value.strip().lower() in ("1", "true", "si", "sí", "yes")
    return cast(value)


class Config:
    # Flask
    SECRET_KEY = _env("WEATHERML_SECRET_KEY", "cambia-esta-clave-en-produccion")
    HOST = _env("WEATHERML_HOST", "127.0.0.1")
    PORT = _env("WEATHERML_PORT", 5001, int)
    DEBUG = _env("WEATHERML_DEBUG", True, bool)
    MAX_CONTENT_LENGTH = _env("WEATHERML_MAX_MB", 10, int) * 1024 * 1024

    # Modelo
    MODEL_PATH = Path(_env("WEATHERML_MODEL_PATH", str(BASE_DIR / "models" / "weather_model.joblib")))
    TARGET = _env("WEATHERML_TARGET", "rain")
    TEST_SIZE = _env("WEATHERML_TEST_SIZE", 0.20, float)
    RANDOM_STATE = _env("WEATHERML_RANDOM_STATE", 42, int)
    N_ESTIMATORS = _env("WEATHERML_N_ESTIMATORS", 100, int)

    # Valores de la columna objetivo que significan "sí llueve"
    POSITIVE_LABELS = ("yes", "si", "sí", "1", "true", "rain", "lluvia")

    # Textos amigables para el formulario de predicción
    FIELD_LABELS = {
        "temperature": "Temperatura",
        "humidity": "Humedad",
        "wind": "Velocidad del viento",
        "cloud": "Nubosidad",
    }
    FIELD_UNITS = {
        "temperature": "°C",
        "humidity": "%",
        "wind": "km/h",
    }
    # Límites físicos permitidos (mínimo, máximo); None = sin límite
    FIELD_LIMITS = {
        "temperature": (-50, 60),
        "humidity": (0, 100),
        "wind": (0, 200),
    }
    OPTION_LABELS = {
        "clear": "Despejado",
        "cloudy": "Nublado",
        "overcast": "Cubierto",
    }
