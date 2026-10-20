from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
class TrainModelFilter:
    """Construye y entrena el modelo de clasificación."""
    def __init__(self, n_estimators=100, random_state=42):
        self.n_estimators = n_estimators
        self.random_state = random_state
    def run(self, X_train, y_train, info):
        model = Pipeline([
            ("preprocessor", info["preprocessor"]),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=self.n_estimators,
                    random_state=self.random_state,
                    class_weight="balanced"
                )
            )
        ])
        model.fit(X_train, y_train)
        return model
