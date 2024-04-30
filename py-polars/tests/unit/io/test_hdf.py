from datetime import datetime
from pathlib import Path

import pytest

import polars as pl
import polars.datatypes.classes as pl_dtypes
from polars.testing import assert_frame_equal, assert_series_equal

@pytest.fixture
def hdf_path(io_files_path: Path) -> Path:
    return io_files_path / "hdf"


def _billionaires_table_data():
    return [
        {"name": "Bernard Arnault", "age": 75, "net_worth_usd": 213.5},
        {"name": "Jeff Bezos", "age": 60, "net_worth_usd": 197.6},
        {"name": "Elon Musk", "age": 52, "net_worth_usd": 191.1},
    ] 


def _billionaires_schema():
    return {"name": pl_dtypes.String, "age": pl_dtypes.Int16, "net_worth_usd": pl_dtypes.Float32}


def _air_quality_table_data():
    return [
        {"timestamp": datetime(year=2024, month=4, day=25).timestamp(), "aqi":412, "city":"Denver", "safe":False},
        {"timestamp": datetime(year=2024, month=4, day=26).timestamp(), "aqi":413, "city":"Denver", "safe":False},
        {"timestamp": datetime(year=2024, month=4, day=25).timestamp(), "aqi":35, "city":"Buenos Aires", "safe":True},
        {"timestamp": datetime(year=2024, month=4, day=26).timestamp(), "aqi":36, "city":"Buenos Aires", "safe":True},
    ]


def _argentina_air_quality_table_data():
    return [
        {"timestamp": datetime(year=2024, month=4, day=25).timestamp(), "aqi":35, "city":"Buenos Aires", "safe":True},
        {"timestamp": datetime(year=2024, month=4, day=26).timestamp(), "aqi":36, "city":"Buenos Aires", "safe":True},
    ]


def _united_states_air_quality_table_data():
    return [
        {"timestamp": datetime(year=2024, month=4, day=25).timestamp(), "aqi":412, "city":"Denver", "safe":False},
        {"timestamp": datetime(year=2024, month=4, day=26).timestamp(), "aqi":413, "city":"Denver", "safe":False},
    ]


def _argentina_ozone_aqi_array_data():
    return [12, 14, 17, 14, 14]


def _united_states_ozone_aqi_array_data():
    return [30, 31, 38, 21, 40]


def _air_quality_schema():
    return {"timestamp": pl_dtypes.Int32, "aqi": pl_dtypes.Int16, "city": pl_dtypes.String, "safe": pl_dtypes.Boolean}


def _read_hdf_test_case(fname, data, id_, schema=None, where=None, slice_=None, **pytables_kwargs):
    return pytest.param(fname, data, schema, where, slice_, pytables_kwargs, id=id_)


@pytest.mark.parametrize("fname,data,schema,where,slice_,pytables_kwargs", [
    _read_hdf_test_case(
        fname="billionaires.h5",
        data=_billionaires_table_data(),
        schema=_billionaires_schema(),
        id_="read_hdf_root_table_01",
    ),
    _read_hdf_test_case(
        fname="air_quality.h5",
        data=_air_quality_table_data(),
        schema=_air_quality_schema(),
        id_="read_hdf_root_table_02",
    ),
    _read_hdf_test_case(
        fname="air_quality.h5",
        data=_argentina_air_quality_table_data(),
        schema=_air_quality_schema(),
        root_uep="/country/ar",
        id_="read_hdf_table_from_nested_group_default_table",
    ),
    _read_hdf_test_case(
        fname="air_quality.h5",
        data=_united_states_air_quality_table_data(),
        schema=_air_quality_schema(),
        title="air_quality",
        root_uep="/country/us",
        id_="read_hdf_table_by_name_from_nested_group",
    ),
    _read_hdf_test_case(
        fname="air_quality.h5",
        data=_argentina_air_quality_table_data(),
        schema=_air_quality_schema(),
        title="air_quality",
        where='city == "Buenos Aires"',
        id_="read_hdf_table_with_a_condition",
    ),
    _read_hdf_test_case(
        fname="air_quality.h5",
        data=_argentina_air_quality_table_data(),
        schema=_air_quality_schema(),
        title="air_quality",
        slice_=slice(2, 4, 1),
        id_="read_hdf_table_with_slicing",
    ),
    # TODO: add more test cases that cover the rest of the supported pytables datatypes
])
def test_read_h5_table(hdf_path, fname, data, schema, where, slice_, pytables_kwargs):
    df = pl.read_hdf(hdf_path / fname, where=where, slice_=slice_, **pytables_kwargs)
    expected = pl.DataFrame(data, schema=schema)
    assert_frame_equal(expected, df)



@pytest.mark.parametrize("fname,data,schema,where,slice_,pytables_kwargs", [
    _read_hdf_test_case(
        fname="air_quality.h5",
        data=_argentina_ozone_aqi_array_data(),
        title="ozone_aqi",
        root_uep="/country/ar",
        id_="read_hdf_array",
    ),
    _read_hdf_test_case(
        fname="air_quality.h5",
        data=_argentina_ozone_aqi_array_data()[::2],
        slice_=slice(None, None, 2),
        title="ozone_aqi",
        root_uep="/country/ar",
        id_="read_hdf_array_with_slicing",
    ),
])
def test_read_h5_array(hdf_path, fname, data, schema, where, slice_, pytables_kwargs):
    s = pl.read_hdf(hdf_path/fname, where=where, slice_=slice_, **pytables_kwargs)
    expected = pl.Series(name="ozone_aqi", values=data)
    assert_series_equal(expected, s)


# TODO: read h5 carray

# TODO: read h5 earray

# TODO: read h5 vlarray

# def test_invalid_file_format():
#     assert False, "Implement this test"


# def test_can_read_from_pandas_pytable_file():
#     assert False, "Implement this test"

# TODO: test compression?
