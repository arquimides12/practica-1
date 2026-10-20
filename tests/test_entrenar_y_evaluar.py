from filtros import (
    EvaluateModelFilter,
    PrepareDataFilter,
    SplitDataFilter,
    TrainModelFilter,
)


def _entrenar(dataframe):
    X, y, info = PrepareDataFilter("rain").run(dataframe)
    X_train, X_test, y_train, y_test = SplitDataFilter().run(X, y)
    model = TrainModelFilter(n_estimators=20).run(X_train, y_train, info)
    return model, X_test, y_test


def test_entrena_modelo_que_predice(dataframe):
    model, X_test, _ = _entrenar(dataframe)
    assert set(model.predict(X_test)) <= {"yes", "no"}


def test_evalua_metricas_en_porcentaje(dataframe):
    model, X_test, y_test = _entrenar(dataframe)
    metrics = EvaluateModelFilter().run(model, X_test, y_test)
    for key in ("accuracy", "precision", "recall", "f1"):
        assert 0 <= metrics[key] <= 100
    assert len(metrics["confusion_matrix"]) == len(metrics["labels"])
