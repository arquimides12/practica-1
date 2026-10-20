import io

from filtros import LoadDatasetFilter


def test_lee_csv_y_devuelve_dataframe():
    df = LoadDatasetFilter().run(io.StringIO("a,b\n1,2\n3,4\n"))
    assert list(df.columns) == ["a", "b"]
    assert len(df) == 2
