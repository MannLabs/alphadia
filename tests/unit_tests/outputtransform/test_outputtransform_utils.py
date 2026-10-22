import os

import pandas as pd
import pytest
from conftest import random_tempfolder

from alphadia.constants.keys import SearchStepFiles
from alphadia.outputtransform.utils import (
    load_psm_files_from_folders,
    read_df,
    write_df,
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
