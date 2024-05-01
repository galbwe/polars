from functools import partial
from pathlib import Path

import tables as tb

import polars.datatypes.classes as pl_dtypes
# TODO: move column mapping functions to a private module
from .functions import _PYTABLES_TO_POLARS_DTYPE_MAPPING


def _write_frame_to_hdf(df: "DataFrame", table: str, path: str | Path | None, group: str | None = None) -> None:
    # TODO: if fpath is None, set fpath to the current working directory
    # TODO: support both writing and appending
    with tb.open_file(path, mode="w") as h5file:
        group_path = group
        if group_path is None:
            group = h5file.root
        else:
            group = _navigate_to_group(h5file, group_path)

        table_description = {}
        for i, x in enumerate(df.schema.items()):
            col, polars_dtype = x
            pytables_column_constructor = _resolve_column_type(polars_dtype)
            table_description[col] = pytables_column_constructor(pos=i)

        table = h5file.create_table(group, table, description=table_description)

        # write rows to table
        columns = tuple(df.schema)  # This allows iterating over tuples instead of dicts, hopefully incurring less memory overhead.
        pytables_row = table.row
        for df_row in df.iter_rows():
            for i, x in enumerate(df_row):
                pytables_row[columns[i]] = x
            pytables_row.append()
        table.flush()


_POLARS_TO_PYTABLES_DTYPE_MAPPING = {
    pl_dtypes.Boolean: tb.BoolCol,
    pl_dtypes.Float32: tb.Float32Col,
    pl_dtypes.Float64: tb.Float64Col,
    pl_dtypes.Int16: tb.Int16Col,
    pl_dtypes.Int32: tb.Int32Col,
    pl_dtypes.Int64: tb.Int64Col,
    pl_dtypes.Int8: tb.Int8Col,
    # TODO: come up with a better way to set itemsize dynamically
    pl_dtypes.String: partial(tb.StringCol, itemsize=1000),
    pl_dtypes.Time: tb.Time64Col(),
    pl_dtypes.UInt16: tb.UInt16Col(),
    pl_dtypes.UInt32: tb.UInt32Col(),
    pl_dtypes.UInt64: tb.UInt64Col(),
    pl_dtypes.UInt8: tb.UInt8Col(),
}

def _resolve_column_type(dtype):
    return _POLARS_TO_PYTABLES_DTYPE_MAPPING[dtype]


def _navigate_to_group(h5file, group_path: str):
    """Traverse from the root of the hdf hierarchy to the requested group.

    Create intermediate groups as needed. 
    """
    if not group_path:
        return None
    path_segments = group_path.split("/")
    # deal with leading forward-slash
    if not path_segments[0] and len(path_segments) > 1:
        path_segments = path_segments[1:]
    # deal with trailing forward-slash
    if not path_segments[-1] and len(path_segments) > 1:
        path_segments = path_segments[:-1]
        
    group = h5file.root
    for segment in path_segments:
        next_group = getattr(group, segment, None)
        if next_group is None:
            next_group = h5file.create_group(group, segment)
        group = next_group

    return group
