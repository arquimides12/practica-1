from sklearn.model_selection import train_test_split


class SplitDataFilter:
    """Separa los datos en entrenamiento y prueba."""

    def __init__(self, test_size=0.20, random_state=42):
        self.test_size = test_size
        self.random_state = random_state

    def run(self, X, y):
        counts = y.value_counts()
        if len(y) < 10:
            raise ValueError(
                "El archivo tiene muy pocos registros. Se necesitan al menos 10."
            )
        # Solo se estratifica si cada clase tiene al menos 2 registros
        stratify = y if counts.min() >= 2 else None
        return train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify,
        )
