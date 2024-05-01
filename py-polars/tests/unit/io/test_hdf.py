from datetime import datetime
from pathlib import Path

import pandas as pd
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


def _air_quality_schema():
    return {"timestamp": pl_dtypes.Int32, "aqi": pl_dtypes.Int16, "city": pl_dtypes.String, "safe": pl_dtypes.Boolean}


def _read_hdf_test_case(fname, data, id_, schema=None, where=None, slice_=None, group=None, leaf=None, **pytables_kwargs):
    return pytest.param(
        fname, 
        data, 
        schema, 
        where, 
        slice_, 
        group, 
        leaf, 
        pytables_kwargs, 
        id=id_)


@pytest.mark.parametrize("fname,data,schema,where,slice_,group,leaf,pytables_kwargs", [
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
        group="/country/ar",
        id_="read_hdf_table_from_nested_group_default_table",
    ),
    _read_hdf_test_case(
        fname="air_quality.h5",
        data=_united_states_air_quality_table_data(),
        schema=_air_quality_schema(),
        leaf="air_quality",
        group="/country/us",
        id_="read_hdf_table_by_name_from_nested_group",
    ),
    _read_hdf_test_case(
        fname="air_quality.h5",
        data=_argentina_air_quality_table_data(),
        schema=_air_quality_schema(),
        leaf="air_quality",
        where='city == "Buenos Aires"',
        id_="read_hdf_table_with_a_condition",
    ),
    _read_hdf_test_case(
        fname="air_quality.h5",
        data=_argentina_air_quality_table_data(),
        schema=_air_quality_schema(),
        leaf="air_quality",
        slice_=slice(2, 4, 1),
        id_="read_hdf_table_with_slicing",
    ),
    # TODO: add more test cases that cover the rest of the supported pytables datatypes
])
def test_read_h5_table(hdf_path, fname, data, schema, where, slice_, group, leaf, pytables_kwargs):
    df = pl.read_hdf(hdf_path / fname, where=where, slice_=slice_, group=group, leaf=leaf, **pytables_kwargs)
    expected = pl.DataFrame(data, schema=schema)
    assert_frame_equal(expected, df)



@pytest.mark.parametrize("fname,data,schema,where,slice_,group,leaf,pytables_kwargs", [
    _read_hdf_test_case(
        fname="air_quality.h5",
        data=_argentina_ozone_aqi_array_data(),
        leaf="ozone_aqi",
        group="/country/ar",
        id_="read_hdf_array",
    ),
    _read_hdf_test_case(
        fname="air_quality.h5",
        data=_argentina_ozone_aqi_array_data()[::2],
        slice_=slice(None, None, 2),
        leaf="ozone_aqi",
        group="/country/ar",
        id_="read_hdf_array_with_slicing",
    ),
])
def test_read_h5_array(hdf_path, fname, data, schema, where, slice_, group, leaf, pytables_kwargs):
    s = pl.read_hdf(hdf_path/fname, group=group, leaf=leaf, where=where, slice_=slice_, **pytables_kwargs)
    expected = pl.Series(name="ozone_aqi", values=data)
    assert_series_equal(expected, s)


# TODO: read h5 carray

# TODO: read h5 earray

# TODO: read h5 vlarray

# def test_invalid_file_format():
#     assert False, "Implement this test"

def test_read_from_h5_file_written_by_pandas(tmp_path):
    dir = tmp_path / "hdf"
    dir.mkdir()
    fpath = dir / "test.h5"

    # create a pandas dataframe and write it to an h5 file
    pandas_df = pd.DataFrame(
    {
        "integer": [1, 2, 3],
        "float": [1.0, 2.0, 3.0],
        "string": ["one", "two", "three"],
    })
    pandas_df.to_hdf(fpath, key="test", mode="w")

    # read the h5 file into a polars df
    df = pl.read_hdf(fpath, group="test")
    expected = pl.DataFrame({
        "integer": [1, 2, 3],
        "float": [1.0, 2.0, 3.0],
        "string": ["one", "two", "three"],
    })
    assert_frame_equal(expected, df)



# TODO: test compression?


def test_write_hdf(tmp_path):
    dir = tmp_path / "hdf"
    dir.mkdir()
    # TODO: test both str and Path objects
    fpath = dir / "test.h5"

    # TODO: add other datatypes
    df = pl.DataFrame({
        "integer": [1, 2, 3],
        "float": [1.0, 2.0, 3.0],
        "string": ["one", "two", "three"],
    })

    table = "test_table"

    df.write_hdf(table, path=fpath)

    # read the dataframe back from the h5 file
    df_read = pl.read_hdf(fpath)
    assert_frame_equal(df, df_read)


def test_write_hdf_to_a_nested_group(tmp_path):
    dir = tmp_path / "hdf"
    dir.mkdir()
    fpath = dir / "test.h5"

    df = pl.DataFrame({
        "x": [1, 2, 3],
        "y": [1.0, 2.0, 3.0],
        "z": ["one", "two", "three"],
    })

    group = "/nested/group"
    table = "test_table"

    df.write_hdf(table, path=fpath, group=group)

    df_read = pl.read_hdf(fpath, group=group, leaf=table)
    assert_frame_equal(df, df_read)


# TODO: test appending to an existing table
