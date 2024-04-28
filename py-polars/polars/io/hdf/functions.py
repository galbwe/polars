from enum import Enum, unique
from pathlib import Path

import tables as tb

import polars.datatypes.classes as pl_dtypes
import polars._reexport as pl
from polars.type_aliases import PolarsDataType


def read_hdf(source: str | Path):
    """"""
    # TODO: better docstring
    # open the file
    with tb.open_file(source, mode="r") as h5file:
        # TODO: check that this is a pytables file
        # TODO: support navigating to a specific table
        # TODO: support filtering the table
        # find the table under the root
        table = next(h5file.root._f_walknodes('Leaf'))
        # get the schema for the table
        schema = {
            col: _resolve_column_dtype(pytables_dtype)
            for col, pytables_dtype in table.coltypes.items()
        }
    
        # get rows for the table
        # TODO: check if the system has enough memory and raise an error if not?
        data = table.read()
        return pl.DataFrame(
            data={
                f: data[f]
                for f in data.dtype.fields
            }, 
            schema=schema
        )


def scan_hdf():
    pass


@unique
class _PytablesDatatype(str, Enum):
    """Enumerates pytables datatypes with supported python equivalents
    
    See https://www.pytables.org/usersguide/datatypes.html for reference 
    """
    BOOL = "bool"
    COMPLEX128 = "complex128"
    COMPLEX64 = "complex64"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    INT16 = "int16"
    INT32 = "int32"
    INT64 = "int64"
    INT8 = "int8"
    STRING = "string"
    TIME32 = "time32"
    TIME64 = "time64"
    UINT16 = "uint16"
    UINT32 = "uint32"
    UINT64 = "uint64"
    UINT8 = "uint8"


_PYTABLES_TO_POLARS_DTYPE_MAPPING = {
    _PytablesDatatype.BOOL: pl_dtypes.Boolean,
    # _PytablesDatatype.COMPLEX128: pl_dtypes.Comp,
    # _PytablesDatatype.COMPLEX64: ,
    _PytablesDatatype.FLOAT32: pl_dtypes.Float32,
    _PytablesDatatype.FLOAT64: pl_dtypes.Float64,
    _PytablesDatatype.INT16: pl_dtypes.Int16,
    _PytablesDatatype.INT32: pl_dtypes.Int32,
    _PytablesDatatype.INT64: pl_dtypes.Int64,
    _PytablesDatatype.INT8: pl_dtypes.Int8,
    _PytablesDatatype.STRING: pl_dtypes.String,
    _PytablesDatatype.TIME32: pl_dtypes.Time,
    _PytablesDatatype.TIME64: pl_dtypes.Time,
    _PytablesDatatype.UINT16: pl_dtypes.UInt16,
    _PytablesDatatype.UINT32: pl_dtypes.UInt32,
    _PytablesDatatype.UINT64: pl_dtypes.UInt64,
    _PytablesDatatype.UINT8: pl_dtypes.UInt8,
}

def _resolve_column_dtype(pytables_dtype: _PytablesDatatype) -> PolarsDataType:
    # TODO: handle KeyError?
    return _PYTABLES_TO_POLARS_DTYPE_MAPPING[pytables_dtype]
