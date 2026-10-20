"""Tubería (pipe) que conecta los filtros.

Cada paso recibe el contexto (un diccionario), ejecuta un filtro y
guarda su salida en el contexto para el siguiente paso.
Para agregar un filtro nuevo solo se añade un Step a la lista de
build_training_pipeline(); app.py no se modifica.
"""
from .carga_dataset import LoadDatasetFilter
from .validar_dataset import ValidateDatasetFilter
from .preparar_datos import PrepareDataFilter
from .dividir_datos import SplitDataFilter
from .entrenar_modelo import TrainModelFilter
from .evaluar_modelo import EvaluateModelFilter


class Step:
    """Conecta un filtro con el contexto: qué lee y qué escribe."""

    def __init__(self, name, filter_, inputs, outputs):
        self.name = name
        self.filter = filter_
        self.inputs = inputs
        self.outputs = outputs

    def __call__(self, context):
        args = [context[key] for key in self.inputs]
        result = self.filter.run(*args)
        if len(self.outputs) == 1:
            result = (result,)
        context.update(zip(self.outputs, result))
        return context


class FilterPipeline:
    def __init__(self, steps):
        self.steps = list(steps)

    def run(self, context):
        for step in self.steps:
            context = step(context)
        return context


def build_training_pipeline(config):
    return FilterPipeline([
        Step("Carga", LoadDatasetFilter(), ["source"], ["raw"]),
        Step("Validación", ValidateDatasetFilter(config.TARGET), ["raw"], ["validated"]),
        Step("Preparación", PrepareDataFilter(config.TARGET), ["validated"], ["X", "y", "info"]),
        Step(
            "División",
            SplitDataFilter(config.TEST_SIZE, config.RANDOM_STATE),
            ["X", "y"],
            ["X_train", "X_test", "y_train", "y_test"],
        ),
        Step(
            "Entrenamiento",
            TrainModelFilter(config.N_ESTIMATORS, config.RANDOM_STATE),
            ["X_train", "y_train", "info"],
            ["model"],
        ),
        Step("Evaluación", EvaluateModelFilter(), ["model", "X_test", "y_test"], ["metrics"]),
    ])
