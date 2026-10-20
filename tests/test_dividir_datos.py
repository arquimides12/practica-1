import pandas as pd
import pytest

from filtros import SplitDataFilter


def test_divide_80_20(dataframe):
    X, y = dataframe.drop(columns=["rain"]), dataframe["rain"]
    X_train, X_test, y_train, y_test = SplitDataFilter(0.2, 42).run(X, y)
    assert len(X_test) == round(len(X) * 0.2)
    assert len(X_train) + len(X_test) == len(X)


def test_rechaza_muy_pocos_registros():
    X = pd.DataFrame({"a": range(5)})
    y = pd.Series(["yes", "no", "yes", "no", "yes"])
    with pytest.raises(ValueError, match="pocos registros"):
        SplitDataFilter().run(X, y)
