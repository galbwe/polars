from enum import Enum, unique
from functools import partial
from pathlib import Path

import tables as tb

import polars.datatypes.classes as pl_dtypes
import polars._reexport as pl
from polars.type_aliases import PolarsDataType


def read_hdf(source: str | Path, group: str | None = None, leaf: str | None = None, where: str | None = None, slice_: slice | None = None, **kwargs):
    """Read a table from a pytables h5 file to a polars dataframe"""
    # TODO: better docstring

    # default to using the source parameter instead of the pytable open_file parameter
    if "filename" in kwargs:
        del kwargs["filename"]
    # only allow reading
    if "mode" in kwargs:
        del kwargs["mode"]
    # use group parameter instead of pytables root_uep
    if "root_uep" in kwargs:
        del kwargs["root_uep"]
    # use leaf parameter instead of pytables title
    if "title" in kwargs:
        del kwargs["title"]

    # set the pytables defaults for group and title
    group = group or "/"
    leaf = leaf or ""

    # open the file
    with tb.open_file(source, mode="r", root_uep=group, title=leaf, **kwargs) as h5file:
        # TODO: check that this is a pytables file
        # TODO: support navigating to a specific table
        # TODO: support filtering the table
        # find the table under the group 

        # look for the correct table in the current group
        leaves = h5file.root._v_leaves.values()
        if leaf:
            # look for the correct node by name
            leaves = [x for x in leaves if x.name == leaf]

        # TODO: handle case where leaf does not exist
        leaf_node = leaves[0]
        # TODO: determine what kind of leaf this is
        leaf_is_array = isinstance(leaf_node, tb.Array)
        if leaf_is_array:
            # convert array data to polars series
            # TODO; support slicing
            data = _read_data(leaf_node, where=None, slice_=slice_)
            dtype = _resolve_column_dtype(str(leaf_node.dtype))
            return pl.Series(name=leaf_node.name, values=data, dtype=dtype)

        # assume leaf is a table

        # get the schema for the table
        schema = {
            col: _resolve_column_dtype(pytables_dtype)
            for col, pytables_dtype in leaf_node.coltypes.items()
        }
    

        # TODO: check if the system has enough memory and raise an error if not?

        data = _read_data(leaf_node, where, slice_)

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


def _read_data(table, where, slice_):
    if where is None:
        read = table.read
    elif isinstance(where, str):
        read = partial(table.read_where, condition=where)
    else:
        raise ValueError(f"read_hdf received invalid where parameter: {where}")
    if slice_ is not None:
        read = partial(read, start=slice_.start, stop=slice_.stop, step=slice_.step)

    return read()
