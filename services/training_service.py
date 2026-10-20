"""Lógica de entrenamiento: ejecuta la tubería de filtros y guarda el modelo."""
import logging

import pandas as pd

from filtros import build_training_pipeline
from .errors import UserError
from .model_store import ModelStore

logger = logging.getLogger(__name__)


class TrainingService:
    def __init__(self, config):
        self.config = config
        self.store = ModelStore(config.MODEL_PATH)

    def train(self, file_storage):
        self._check_file(file_storage)
        pipeline = build_training_pipeline(self.config)
        try:
            ctx = pipeline.run({"source": file_storage.stream})
        except UserError:
            raise
        except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError):
            raise UserError("No pudimos leer el archivo. Verifica que sea un CSV válido.")
        except ValueError as exc:
            # Los filtros lanzan ValueError con mensajes ya pensados para el usuario
            raise UserError(str(exc))
        except Exception:
            logger.exception("Error inesperado durante el entrenamiento")
            raise UserError(
                "Ocurrió un problema al entrenar el modelo. Revisa el archivo e inténtalo de nuevo."
            )

        X, y, metrics = ctx["X"], ctx["y"], ctx["metrics"]
        results = {
            "rows": len(ctx["validated"]),
            "features": len(X.columns),
            "target": ctx["info"]["target"],
            "class_distribution": {str(k): int(v) for k, v in y.value_counts().items()},
            "metrics": metrics,
            "model_path": self.config.MODEL_PATH.name,
        }
        self.store.save({
            "model": ctx["model"],
            "features": list(X.columns),
            "target": ctx["info"]["target"],
            "schema": self.build_schema(X, ctx["info"]),
            "results": results,
        })
        return results

    @staticmethod
    def _check_file(file_storage):
        if file_storage is None or not file_storage.filename:
            raise UserError("Selecciona un archivo CSV.")
        if not file_storage.filename.lower().endswith(".csv"):
            raise UserError("El archivo debe tener extensión .csv.")

    @staticmethod
    def build_schema(X, info):
        """Describe cada campo del formulario: número (con rango) o lista."""
        schema = []
        for column in X.columns:
            if column in info["numeric_columns"]:
                values = pd.to_numeric(X[column], errors="coerce").dropna()
                schema.append({
                    "name": column,
                    "type": "number",
                    "min": round(float(values.min()), 2),
                    "max": round(float(values.max()), 2),
                    "default": round(float(values.median()), 2),
                })
            else:
                options = sorted(X[column].dropna().astype(str).str.strip().unique())
                schema.append({"name": column, "type": "select", "options": options})
        return schema
