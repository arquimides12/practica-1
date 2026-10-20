"""Lógica de predicción: arma el formulario, valida los datos y predice."""
import logging

import pandas as pd

from .errors import UserError
from .model_store import ModelStore

logger = logging.getLogger(__name__)


class PredictionService:
    def __init__(self, config):
        self.config = config
        self.store = ModelStore(config.MODEL_PATH)

    def has_model(self):
        return self.store.exists()

    def _load(self):
        if not self.has_model():
            raise UserError("Primero entrena un modelo subiendo un archivo CSV.")
        return self.store.load()

    def last_results(self):
        return self._load().get("results")

    def form_fields(self):
        """Campos del formulario con etiquetas, unidades y opciones en español."""
        saved = self._load()
        schema = saved.get("schema") or self._schema_from_model(saved)
        fields = []
        for field in schema:
            item = dict(field)
            item["label"] = self.config.FIELD_LABELS.get(field["name"], field["name"].capitalize())
            item["unit"] = self.config.FIELD_UNITS.get(field["name"], "")
            item["min_allowed"], item["max_allowed"] = self.config.FIELD_LIMITS.get(
                field["name"], (None, None)
            )
            if field["type"] == "select":
                item["options"] = [
                    {"value": o, "label": self.config.OPTION_LABELS.get(o, o.capitalize())}
                    for o in field["options"]
                ]
            fields.append(item)
        return fields

    def predict(self, form):
        saved = self._load()
        fields = self.form_fields()
        values = {}
        for field in fields:
            raw = (form.get(field["name"]) or "").strip()
            if raw == "":
                raise UserError(f"Completa el campo «{field['label']}».")
            if field["type"] == "number":
                try:
                    number = float(raw.replace(",", "."))
                except ValueError:
                    raise UserError(f"«{field['label']}» debe ser un número.")
                if field.get("min_allowed") is not None and number < field["min_allowed"]:
                    raise UserError(f"«{field['label']}» no puede ser menor que {field['min_allowed']}.")
                if field.get("max_allowed") is not None and number > field["max_allowed"]:
                    raise UserError(f"«{field['label']}» no puede ser mayor que {field['max_allowed']}.")
                values[field["name"]] = number
            else:
                valid = [o["value"] for o in field["options"]]
                if raw not in valid:
                    raise UserError(f"Elige una opción válida en «{field['label']}».")
                values[field["name"]] = raw

        try:
            model = saved["model"]
            sample = pd.DataFrame([values], columns=saved["features"])
            label = str(model.predict(sample)[0])
            confidence = None
            if hasattr(model, "predict_proba"):
                confidence = round(float(max(model.predict_proba(sample)[0])) * 100, 1)
        except Exception:
            logger.exception("Error inesperado durante la predicción")
            raise UserError("No se pudo generar la predicción. Inténtalo de nuevo.")

        will_rain = label.strip().lower() in self.config.POSITIVE_LABELS
        return {
            "will_rain": will_rain,
            "text": "Sí lloverá" if will_rain else "No lloverá",
            "confidence": confidence,
            "label": label,
            "inputs": {k: form.get(k) for k in values},
        }

    @staticmethod
    def _schema_from_model(saved):
        """Para modelos antiguos sin esquema: lo deduce del preprocesador."""
        pre = saved["model"].named_steps["preprocessor"]
        schema = []
        for name, transformer, columns in pre.transformers_:
            if name == "numeric":
                medians = transformer.named_steps["imputer"].statistics_
                for column, median in zip(columns, medians):
                    schema.append({"name": column, "type": "number", "min": None,
                                   "max": None, "default": round(float(median), 2)})
            elif name == "categorical":
                categories = transformer.named_steps["encoder"].categories_
                for column, options in zip(columns, categories):
                    schema.append({"name": column, "type": "select",
                                   "options": [str(o) for o in options]})
        order = {c: i for i, c in enumerate(saved["features"])}
        return sorted(schema, key=lambda f: order.get(f["name"], 0))
