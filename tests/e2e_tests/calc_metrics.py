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
base_path = sys.argv[1]

output_dir = "/output/"
input_file_name = "stat.tsv"

# -
print("reading from", base_path + output_dir + input_file_name)

df = pd.read_csv(base_path + output_dir + input_file_name, sep="\t")

# +

run = neptune.init_run(
    project=neptune_project,
)

run["e2e-test"] = "jahuu" + base_path
# run["config"] = # TODO add config
# run["config"] = # TODO add more metatdate: commit hash, ...

try:
    for col in ["proteins", "precursors", "ms1_accuracy", "fwhm_rt"]:
        run[f"stat/{col}_mean"] = df[col].mean()
        run[f"stat/{col}_std"] = df[col].std()
except Exception:
    pass
finally:
    run.stop()

# -
