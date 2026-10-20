import pandas as pd
import pytest

from filtros import ValidateDatasetFilter


def test_rechaza_dataset_vacio():
    with pytest.raises(ValueError, match="vacío"):
        ValidateDatasetFilter().run(pd.DataFrame())


def test_rechaza_si_falta_columna_objetivo():
    with pytest.raises(ValueError, match="rain"):
        ValidateDatasetFilter().run(pd.DataFrame({"humidity": [1, 2]}))


def test_elimina_duplicados_y_nulos_en_objetivo():
    df = pd.DataFrame({"humidity": [1, 1, 2, 3], "rain": ["yes", "yes", "no", None]})
    result = ValidateDatasetFilter().run(df)
    assert len(result) == 2


def test_exige_dos_clases():
    with pytest.raises(ValueError, match="dos clases"):
        ValidateDatasetFilter().run(pd.DataFrame({"humidity": [1, 2], "rain": ["yes", "yes"]}))
