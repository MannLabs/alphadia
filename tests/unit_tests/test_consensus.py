import pytest
import pandas as pd
import os
from conftest import random_tempfolder
from alphadia.consensus.utils import read_df, write_df


@pytest.mark.parametrize(
    "format, should_fail",
    [("tsv", False), ("parquet", False), ("a321", True)],
)
def test_read_write(format, should_fail):
    # given
    df = pd.DataFrame([{"a": "a", "b": "b"}, {"a": "a", "b": "b"}])
    path = os.path.join(random_tempfolder())

    # when
    if should_fail:
        with pytest.raises(ValueError):
            write_df(df, path, file_format=format)

    else:
        write_df(df, path, file_format=format)
        _df = read_df(path, file_format=format)
        assert df.equals(_df)
