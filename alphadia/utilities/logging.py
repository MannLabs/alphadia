import logging
import os
import socket
from datetime import datetime
from importlib import metadata

import alphabase
import alpharaw
import alphatims
import directlfq
import peptdeep

import alphadia
from alphadia.utils import USE_NUMBA_CACHING

logger = logging.getLogger()


def print_logo() -> None:
    """Print the alphadia logo and version."""
    logger.progress("          _      _         ___ ___   _   ")
    logger.progress(r"     __ _| |_ __| |_  __ _|   \_ _| /_\  ")
    logger.progress("    / _` | | '_ \\ ' \\/ _` | |) | | / _ \\ ")
    logger.progress("    \\__,_|_| .__/_||_\\__,_|___/___/_/ \\_\\")
    logger.progress("           |_|                           ")
    logger.progress("")
    logger.progress(f"version: {alphadia.__version__}")


def print_environment() -> None:
    """Log information about the python environment."""

    logger.info(f"hostname: {socket.gethostname()}")
    if slurm_job_id := os.environ.get("SLURM_JOB_ID"):
        logger.info(f"slurm_job_id: {slurm_job_id}")

    now = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"date: {now}")

    logger.info("================ AlphaX Environment ===============")
    logger.info(f"{'alphatims':<15} : {alphatims.__version__:}")
    logger.info(f"{'alpharaw':<15} : {alpharaw.__version__}")
    logger.info(f"{'alphabase':<15} : {alphabase.__version__}")
    logger.info(f"{'alphapeptdeep':<15} : {peptdeep.__version__}")
    logger.info(f"{'directlfq':<15} : {directlfq.__version__}")
    logger.info("===================================================")

    logger.info("================= Pip Environment =================")
    pip_env = [
        f"{dist.metadata['Name']}=={dist.version}" for dist in metadata.distributions()
    ]
    logger.info(" ".join(pip_env))
    logger.info("===================================================")

    if USE_NUMBA_CACHING:
        logger.info("Numba caching is activated.")
