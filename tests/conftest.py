import io
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import Config  # noqa: E402

DATA = ROOT / "data" / "ejemplo_clima.csv"


@pytest.fixture
def dataframe():
    return pd.read_csv(DATA)


@pytest.fixture
def csv_bytes():
    return DATA.read_bytes()


@pytest.fixture
def test_config(tmp_path):
    class TestConfig(Config):
        MODEL_PATH = tmp_path / "modelo_prueba.joblib"
        N_ESTIMATORS = 20
        TESTING = True
    return TestConfig


@pytest.fixture
def client(test_config):
    from app import create_app
    app = create_app(test_config)
    return app.test_client()


def upload(client, content, filename="datos.csv"):
    return client.post(
        "/train",
        data={"dataset": (io.BytesIO(content), filename)},
        content_type="multipart/form-data",
    )
