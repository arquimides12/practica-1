"""Guarda y lee el modelo entrenado desde disco."""
import joblib


class ModelStore:
    def __init__(self, path):
        self.path = path

    def exists(self):
        return self.path.exists()

    def save(self, bundle):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, self.path)

    def load(self):
        return joblib.load(self.path)
