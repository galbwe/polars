from datetime import datetime
from pathlib import Path

import pytest

import polars as pl
import polars.datatypes.classes as pl_dtypes
from polars.testing import assert_frame_equal

@pytest.fixture
def hdf_path(io_files_path: Path) -> Path:
    return io_files_path / "hdf"


@pytest.mark.parametrize("fname,data,schema", [
    pytest.param(
        # fname
        "billionaires.h5",
        # data
        [
          {"name": "Bernard Arnault", "age": 75, "net_worth_usd": 213.5},
          {"name": "Jeff Bezos", "age": 60, "net_worth_usd": 197.6},
          {"name": "Elon Musk", "age": 52, "net_worth_usd": 191.1},
        ],
        # schema
        {"name": pl_dtypes.String, "age": pl_dtypes.Int16, "net_worth_usd": pl_dtypes.Float32},
        id="read_small_table_from_root_group_1",
    ),
    pytest.param(
        # fname
        "air_quality.h5",
        # data
        [
            {"timestamp": datetime(year=2024, month=4, day=25).timestamp(), "aqi":412, "city":"Denver", "safe":False},
            {"timestamp": datetime(year=2024, month=4, day=26).timestamp(), "aqi":413, "city":"Denver", "safe":False},
            {"timestamp": datetime(year=2024, month=4, day=25).timestamp(), "aqi":35, "city":"Buenos Aires", "safe":True},
            {"timestamp": datetime(year=2024, month=4, day=26).timestamp(), "aqi":36, "city":"Buenos Aires", "safe":True},
        ],
        # schema
        {"timestamp": pl_dtypes.Int32, "aqi": pl_dtypes.Int16, "city": pl_dtypes.String, "safe": pl_dtypes.Boolean},
        id="read_small_table_from_root_group_2",
    ),
    # TODO: add more test cases that cover the rest of the supported pytables datatypes
])
def test_read_h5_table_from_root_group(hdf_path, fname, data, schema):
    df = pl.read_hdf(hdf_path / fname)
    expected = pl.DataFrame(data, schema=schema)
    assert_frame_equal(expected, df)


def test_invalid_file_format():
    assert False, "Implement this test"


def test_can_read_from_pandas_pytable_file():
    assert False, "Implement this test"
