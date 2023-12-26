# setup basic argpasrse comand line script to run the fdr test
# with the options --threads and --size between 0 and 100%

import matplotlib
import argparse
import tempfile
import pandas as pd
import torch
import numpy as np
import time
import neptune

import alphadia
from alphadia import testing, fdr, fdrexperimental
from alphadia.workflow import peptidecentric

classifiers = {
    "binary": fdrexperimental.BinaryClassifier,
    "legacy": fdrexperimental.BinaryClassifierLegacy,
    "legacy_new_batching": fdrexperimental.BinaryClassifierLegacyNewBatching,
}

parser = argparse.ArgumentParser(description="Run the fdr test.")
parser.add_argument(
    "--threads", type=int, default=1, help="number of threads to use, default: 1"
)
parser.add_argument(
    "--size", type=int, default=100, help="size of the test in percent, default: 100"
)

parser.add_argument(
    "--url",
    type=str,
    default="https://datashare.biochem.mpg.de/s/aGpiFGNP1CcljTY",
    help="url to the test data, default: https://datashare.biochem.mpg.de/s/aGpiFGNP1CcljTY",
)

parser.add_argument(
    "--neptune-project",
    type=str,
    default="MannLabs/alphadia-fdr-optimization",
    help="Neptune.ai project for continous logging, default: MannLabs/alphadia-fdr-optimization",
)

parser.add_argument(
    "--neptune-tag",
    action="append",
    dest="neptune_tag",
    default=[],
    help="Specify Neptune tags",
)

parser.add_argument(
    "--n-iter", type=int, default=20, help="number of iterations, default: 20"
)

parser.add_argument(
    "--max-batch-size",
    type=int,
    default=10000,
    help="maximum batch size, default: 10000",
)

parser.add_argument(
    "--min-batch-number",
    type=int,
    default=100,
    help="minimum batch number, default: 100",
)

parser.add_argument(
    "--epochs", type=int, default=10, help="number of epochs, default: 10"
)

parser.add_argument(
    "--learning-rate", type=float, default=0.0002, help="learning rate, default: 0.0002"
)

parser.add_argument(
    "--weight-decay", type=float, default=0.00001, help="weight decay, default: 0.00001"
)

parser.add_argument(
    "--layers",
    type=str,
    default="100,50,20,5",
    help="number of layers, default: 100,50,20,5",
)

parser.add_argument(
    "--dropout", type=float, default=0.001, help="dropout, default: 0.001"
)

parser.add_argument(
    "--classifier",
    choices=classifiers.keys(),
    default="binary",
    help="classifier to use, default: binary",
)


def main():
    # disable interactive plotting
    matplotlib.use("Agg")

    # parse command line arguments
    args = parser.parse_args()

    # print all command line arguments
    print("==========================")
    print("Command line arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    # set number of threads
    torch.set_num_threads(args.threads)

    # set neptune.ai logging
    run = neptune.init_run(
        project=args.neptune_project,
        tags=args.neptune_tag,
    )

    # log parameters to neptune.ai
    run["parameters"] = vars(args)
    run["version"] = alphadia.__version__

    print(f"Downloading test data from {args.url}...")
    temp_directory = tempfile.gettempdir()
    test_data_location = testing.update_datashare(args.url, temp_directory)
    print(f"Saved test data to {test_data_location}")

    features_df = pd.read_csv(test_data_location, sep="\t")
    print(f"Test data has {len(features_df)} rows")
    print(f"Will use {args.size}% of the data")
    available_columns = list(
        set(peptidecentric.feature_columns).intersection(set(features_df.columns))
    )

    performance_dicts = []

    for iteration in range(args.n_iter):
        print(f"Iteration {iteration+1}/{args.n_iter}")

        target_df = features_df[features_df["decoy"] == 0].sample(frac=args.size / 100)
        decoy_df = features_df[features_df["decoy"] == 1].sample(frac=args.size / 100)

        start_time = time.time()

        classifier = classifiers[args.classifier](
            test_size=0.001,
            max_batch_size=args.max_batch_size,
            min_batch_number=args.min_batch_number,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            layers=[int(l) for l in args.layers.split(",")],
            dropout=args.dropout,
        )

        psm_df = fdr.perform_fdr(
            classifier,
            available_columns,
            target_df,
            decoy_df,
            competetive=True,
            neptune_run=run,
        )

        stop_time = time.time()
        duration = stop_time - start_time

        psm_sig_df = psm_df[psm_df["decoy"] == 0]

        performance_dicts.append(
            {
                "duration": duration,
                "iteration": iteration,
                "fdr1": np.sum(psm_sig_df["qval"] < 0.01),
                "fdr01": np.sum(psm_sig_df["qval"] < 0.001),
            }
        )

    performance_df = pd.DataFrame(performance_dicts)
    performance_df["fdr_ratio"] = performance_df["fdr01"] / performance_df["fdr1"]

    run["eval/fdr1_mean"] = performance_df["fdr1"].mean()
    run["eval/fdr1_std"] = performance_df["fdr1"].std()
    run["eval/fdr1_min"] = performance_df["fdr1"].min()
    run["eval/fdr1_max"] = performance_df["fdr1"].max()
    run["eval/fdr1_iszero"] = (performance_df["fdr1"] == 0).sum()

    run["eval/fdr01_mean"] = performance_df["fdr01"].mean()
    run["eval/fdr01_std"] = performance_df["fdr01"].std()
    run["eval/fdr01_min"] = performance_df["fdr01"].min()
    run["eval/fdr01_max"] = performance_df["fdr01"].max()
    run["eval/fdr01_iszero"] = (performance_df["fdr01"] == 0).sum()

    run["eval/fdr_ratio_mean"] = performance_df["fdr_ratio"].mean()
    run["eval/fdr_ratio_std"] = performance_df["fdr_ratio"].std()
    run["eval/fdr_ratio_min"] = performance_df["fdr_ratio"].min()
    run["eval/fdr_ratio_max"] = performance_df["fdr_ratio"].max()

    run["eval/duration_mean"] = performance_df["duration"].mean()
    run["eval/duration_std"] = performance_df["duration"].std()
    run["eval/duration_min"] = performance_df["duration"].min()
    run["eval/duration_max"] = performance_df["duration"].max()

    run.stop()

    print("==========================")
    print("Precursor @ 1% FDR")
    print(
        f'mean: {performance_df["fdr1"].mean()}, std: {performance_df["fdr1"].std():.2f}'
    )
    print(
        f'min: {performance_df["fdr1"].min()}, max: {performance_df["fdr1"].max():.2f}'
    )
    print("")
    print("Precursor @ 0.1% FDR")
    print(
        f'mean: {performance_df["fdr01"].mean()}, std: {performance_df["fdr01"].std():.2f}'
    )
    print(
        f'min: {performance_df["fdr01"].min()}, max: {performance_df["fdr01"].max():.2f}'
    )
    print("")
    print("FDR ratio")
    print(
        f'mean: {performance_df["fdr_ratio"].mean()}, std: {performance_df["fdr_ratio"].std():.2f}'
    )
    print(
        f'min: {performance_df["fdr_ratio"].min()}, max: {performance_df["fdr_ratio"].max():.2f}'
    )
    print("==========================")


if __name__ == "__main__":
    main()
