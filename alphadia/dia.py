#!python


# builtin
import logging

# import external
import alphatims.bruker
import numpy as np

# import local
import alphadia.library
import alphadia.thermo


def run_analysis(
    dia_file_name: str,
    alphapept_library_file_name: str,
    output_file_name: str,
    ppm: float,
    rt_tolerance: float,
    mobility_tolerance: float,
    max_scan_difference: int,
    max_cycle_difference: int,
    thread_count: int,
    fdr_rate: float,
):
    alphatims.utils.set_threads(thread_count)
    logging.info("Loading DIA data")
    if dia_file_name.endswith(".raw"):
        dia_data = alphadia.thermo.RawFile(dia_file_name)
    else:
        dia_data = alphatims.bruker.TimsTOF(dia_file_name)
    logging.info("Loading target library")
    library = alphadia.library.Library(
        alphapept_library_file_name,
        decoy=False
    )
    logging.info(f"Target library contains {len(library)} targets")
    logging.info("Loading decoy library")
    decoy_library = alphadia.library.Library(
        alphapept_library_file_name,
        decoy=True
    )
    logging.info(f"Decoy library contains {len(decoy_library)} decoys")
    logging.info("Scoring target library")
    scores_df = library.score(
        dia_data,
        max_scan_difference=max_scan_difference,
        max_cycle_difference=max_cycle_difference,
        ppm=ppm,
        rt_tolerance=rt_tolerance,  # seconds
        mobility_tolerance=mobility_tolerance,  # 1/k0
    )
    logging.info("Scoring decoy library")
    decoy_scores_df = decoy_library.score(
        dia_data,
        max_scan_difference=max_scan_difference,
        max_cycle_difference=max_cycle_difference,
        ppm=ppm,
        rt_tolerance=rt_tolerance,  # seconds
        mobility_tolerance=mobility_tolerance,  # 1/k0
    )
    logging.info("Calculating FDR")
    fdr_df = alphadia.library.train_and_score(
        scores_df,
        decoy_scores_df,
        train_fdr_level=0.5,
        # min_train=100,
        # plot=True,
    )
    reachable = np.sum(fdr_df.target)
    hits = np.sum((fdr_df['q_value'] <= fdr_rate) & fdr_df.target.values)
    hit_rate = hits / reachable
    logging.info(
        f"Found {reachable} reachable targets"
    )
    logging.info(
        f"Found {hits} ({100 * hit_rate:.2f}%) targets at FDR {fdr_rate}"
    )
    logging.info("Exporting results")
    fdr_df.to_csv(output_file_name)
