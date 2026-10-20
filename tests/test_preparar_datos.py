from filtros import PrepareDataFilter


def test_separa_x_y_e_identifica_tipos(dataframe):
    X, y, info = PrepareDataFilter("rain").run(dataframe)
    assert "rain" not in X.columns
    assert len(X) == len(y)
    assert info["numeric_columns"] == ["temperature", "humidity", "wind"]
    assert info["categorical_columns"] == ["cloud"]
    assert info["preprocessor"] is not None
