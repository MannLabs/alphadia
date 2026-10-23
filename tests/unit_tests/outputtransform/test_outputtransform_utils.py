import os

import pandas as pd
import pytest
from conftest import random_tempfolder

from alphadia.constants.keys import SearchStepFiles
from alphadia.outputtransform.utils import (
    load_psm_files_from_folders,
    read_df,
    write_df,
    write_table,
)


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


def test_write_df_raises_for_supported_format_without_writer(monkeypatch):
    # given
    df = pd.DataFrame({"a": [1.0, 2.0]})
    path = os.path.join(random_tempfolder(), "table")
    monkeypatch.setattr(
        "alphadia.outputtransform.utils.supported_formats", ["parquet", "tsv", "csv"]
    )

    # when / then
    with pytest.raises(ValueError):
        write_df(df, path, file_format="csv")


@pytest.mark.parametrize("file_format", ["tsv", "parquet"])
def test_write_table_infers_format_from_file_name(file_format):
    # given
    df = pd.DataFrame({"a": [1.0, 2.0]})
    path_no_format = os.path.join(random_tempfolder(), "stats.transfer")

    # when
    write_table(df, f"{path_no_format}.{file_format}")

    # then
    assert read_df(path_no_format, file_format=file_format).equals(df)


def test_write_table_rejects_unknown_suffix():
    # given
    df = pd.DataFrame({"a": [1.0, 2.0]})
    file_path = os.path.join(random_tempfolder(), "stat.csv")

    # when / then
    with pytest.raises(ValueError, match="csv"):
        write_table(df, file_path)


def test_load_psm_files_from_folders_reads_psm_file_and_skips_missing():
    # given
    folder_with_psm = random_tempfolder()
    folder_without_psm = random_tempfolder()
    psm_df = pd.DataFrame({"precursor_idx": [1, 2]})
    psm_df.to_parquet(
        os.path.join(folder_with_psm, SearchStepFiles.PSM_FILE_NAME), index=False
    )

    # when
    psm_df_list = load_psm_files_from_folders([folder_with_psm, folder_without_psm])

    # then
    assert len(psm_df_list) == 1
    assert psm_df_list[0].equals(psm_df)
