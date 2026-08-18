"""Probes that make a silent interpreter exit visible.

alphaDIA catches `Exception` at every level, so a `SystemExit` raised anywhere
below passes through unreported and ends the process with its own exit code,
producing no traceback and no log. Together with faulthandler these probes tell
apart a `sys.exit()`, a crash in a C extension, and a hard kill that no handler
can observe.
"""

import atexit
import faulthandler
import logging
import traceback

logger = logging.getLogger()

# mutable so the atexit handler observes updates without a global statement
_state = {"finished": False}


def enable_exit_probes() -> None:
    """Install the probes, as early in the process as possible."""
    faulthandler.enable()
    atexit.register(_report_unfinished_exit)


def mark_finished() -> None:
    """Record that the run reached its end, so the exit probe stays quiet."""
    _state["finished"] = True


def log_system_exit(e: SystemExit) -> None:
    """Log a SystemExit, which no `except Exception` in alphaDIA reports."""
    logger.error(f"SystemExit with code {e.code}, raised at:\n{traceback.format_exc()}")


def _report_unfinished_exit() -> None:
    if not _state["finished"]:
        logger.error(
            "Interpreter is exiting before alphaDIA finished, without an exception. "
            "A hard kill or a crash in a C extension would not have reached this."
        )
