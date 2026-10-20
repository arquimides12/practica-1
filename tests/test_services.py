import io

import pytest
from werkzeug.datastructures import FileStorage

from services import PredictionService, TrainingService, UserError


def _file(content, name="datos.csv"):
    return FileStorage(stream=io.BytesIO(content), filename=name)


def test_entrena_y_guarda_modelo(test_config, csv_bytes):
    results = TrainingService(test_config).train(_file(csv_bytes))
    assert test_config.MODEL_PATH.exists()
    assert results["rows"] > 0


@pytest.mark.parametrize("content,name,message", [
    (b"a,b\n1,2\n", "datos.txt", ".csv"),
    (b"", "datos.csv", "CSV válido"),
    (b"temperature,humidity\n20,70\n", "datos.csv", "rain"),
])
def test_errores_con_mensaje_simple(test_config, content, name, message):
    with pytest.raises(UserError, match=message):
        TrainingService(test_config).train(_file(content, name))


def test_prediccion_devuelve_si_o_no_con_confianza(test_config, csv_bytes):
    TrainingService(test_config).train(_file(csv_bytes))
    result = PredictionService(test_config).predict(
        {"temperature": "18", "humidity": "95", "wind": "25", "cloud": "overcast"}
    )
    assert result["text"] in ("Sí lloverá", "No lloverá")
    assert 0 < result["confidence"] <= 100


def test_formulario_tiene_numeros_y_lista(test_config, csv_bytes):
    TrainingService(test_config).train(_file(csv_bytes))
    fields = {f["name"]: f for f in PredictionService(test_config).form_fields()}
    assert fields["humidity"]["type"] == "number"
    assert fields["cloud"]["type"] == "select"
    assert {o["label"] for o in fields["cloud"]["options"]} == {"Despejado", "Nublado", "Cubierto"}


@pytest.mark.parametrize("form,message", [
    ({"temperature": "", "humidity": "70", "wind": "5", "cloud": "clear"}, "Completa"),
    ({"temperature": "abc", "humidity": "70", "wind": "5", "cloud": "clear"}, "número"),
    ({"temperature": "20", "humidity": "150", "wind": "5", "cloud": "clear"}, "mayor que 100"),
    ({"temperature": "20", "humidity": "70", "wind": "5", "cloud": "nieve"}, "opción válida"),
])
def test_prediccion_valida_los_campos(test_config, csv_bytes, form, message):
    TrainingService(test_config).train(_file(csv_bytes))
    with pytest.raises(UserError, match=message):
        PredictionService(test_config).predict(form)
