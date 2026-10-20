from tests.conftest import upload


def test_inicio_muestra_carga(client):
    html = client.get("/").get_data(as_text=True)
    assert "Cargar datos" in html


def test_entrenar_muestra_metricas_y_formulario(client, csv_bytes):
    html = upload(client, csv_bytes).get_data(as_text=True)
    assert "Matriz de confusión" in html
    assert 'id="prediccion"' in html
    assert 'type="number"' in html and "<select" in html


def test_predecir_muestra_resultado(client, csv_bytes):
    upload(client, csv_bytes)
    html = client.post("/predict", data={
        "temperature": "28", "humidity": "40", "wind": "5", "cloud": "clear"
    }).get_data(as_text=True)
    assert "lloverá" in html and "Confianza del modelo" in html


def test_error_sin_texto_tecnico(client):
    response = upload(client, b"temperature,humidity\n20,70\n")
    html = response.get_data(as_text=True)
    assert response.status_code == 400
    assert "rain" in html and "Traceback" not in html
