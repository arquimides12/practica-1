import io

from config import Config
from filtros import FilterPipeline, Step, build_training_pipeline


class DoubleFilter:
    def run(self, value):
        return value * 2


def test_pipeline_ejecuta_los_pasos_en_orden():
    pipeline = FilterPipeline([
        Step("x2", DoubleFilter(), ["n"], ["n"]),
        Step("x2", DoubleFilter(), ["n"], ["n"]),
    ])
    assert pipeline.run({"n": 3})["n"] == 12


def test_agregar_filtro_es_solo_agregarlo_a_la_lista():
    pipeline = build_training_pipeline(Config)
    pipeline.steps.append(Step("Extra", DoubleFilter(), ["n"], ["n"]))
    assert [s.name for s in pipeline.steps][-1] == "Extra"


def test_pipeline_completo_produce_modelo_y_metricas(csv_bytes):
    ctx = build_training_pipeline(Config).run({"source": io.BytesIO(csv_bytes)})
    assert "model" in ctx and "metrics" in ctx
