# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: alphadia
#     language: python
#     name: alphadia
# ---
import sys

import pandas as pd
import neptune

# TODO take from github
neptune_project = "MannLabs/alphaDIA-e2e-tests"

# +
test_case_name = sys.argv[1]

output_dir = "/output/"
input_file_name = "stat.tsv"

# -
print("reading from", test_case_name + output_dir + input_file_name)


# +

run = neptune.init_run(project=neptune_project, tags=test_case_name)

run["e2e-test"] = test_case_name
# run["config"] = # TODO add config
# run["config"] = # TODO add more metatdate: commit hash, ...

try:
    df = pd.read_csv(test_case_name + output_dir + input_file_name, sep="\t")

    for col in ["proteins", "precursors", "ms1_accuracy", "fwhm_rt"]:
        run[f"stat/{col}_mean"] = df[col].mean()
        run[f"stat/{col}_std"] = df[col].std()
except Exception:
    pass
finally:
    run.stop()

# -
