"""Probes that make a silent interpreter exit visible.

alphaDIA catches `Exception` at every level, so a `SystemExit` raised anywhere
below passes through unreported and ends the process with its own exit code,
producing no traceback and no log. Together with faulthandler and the signal
handlers these probes tell apart a `sys.exit()`, a crash in a C extension, an
external termination signal such as the SIGTERM a scheduler sends on a time or
memory limit, and a SIGKILL that nothing can observe.
"""

import atexit
import faulthandler
import logging
import os
import signal
import traceback
from types import FrameType

logger = logging.getLogger()

# termination signals a scheduler may send. SIGKILL is deliberately absent as it
# cannot be caught. SIGINT is left alone so Ctrl-C keeps raising KeyboardInterrupt.
_TERMINATION_SIGNAL_NAMES = ("SIGTERM", "SIGXCPU", "SIGUSR1", "SIGUSR2", "SIGHUP")

# mutable so the atexit handler observes updates without a global statement
_state = {"finished": False}


def enable_exit_probes() -> None:
    """Install the probes, as early in the process as possible."""
    faulthandler.enable()
    atexit.register(_report_unfinished_exit)

    _install_termination_handlers()

    # multiprocessing pools, such as the one directLFQ uses, SIGTERM their
    # workers as part of normal teardown. Forked children inherit the handlers
    # above, so they must be reset or that teardown is reported as a failure.
    os.register_at_fork(after_in_child=_reset_termination_handlers)


def _install_termination_handlers() -> None:
    """Report and stack dump on signals that terminate the process."""
    for signum in _termination_signals():
        signal.signal(signum, _on_termination_signal)
        if hasattr(faulthandler, "register"):
            # dumps every thread from the C handler, which runs even while the
            # interpreter is blocked inside a native call
            faulthandler.register(signum, chain=True)


def _reset_termination_handlers() -> None:
    """Restore the default handling a process has without these probes."""
    for signum in _termination_signals():
        signal.signal(signum, signal.SIG_DFL)
        if hasattr(faulthandler, "unregister"):
            faulthandler.unregister(signum)

    _state["finished"] = True


def _termination_signals() -> list[int]:
    """Return the termination signals available on this platform."""
    return [
        getattr(signal, name)
        for name in _TERMINATION_SIGNAL_NAMES
        if hasattr(signal, name)
    ]


def _on_termination_signal(signum: int, _frame: FrameType | None) -> None:
    """Report an external termination, then die from that same signal."""
    logger.error(
        f"Received {signal.Signals(signum).name} from outside the process; "
        f"alphaDIA did not choose to exit. A scheduler sends this on a time or "
        f"memory limit, or on a cancel request."
    )
    logging.shutdown()

    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


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
