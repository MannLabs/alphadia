import logging, os, time, datetime
PROGRESS_LEVELV_NUM = 100

logging.PROGRESS = PROGRESS_LEVELV_NUM
logging.addLevelName(PROGRESS_LEVELV_NUM, "PROGRESS")
def progress(self, message, *args, **kws):
    if self.isEnabledFor(PROGRESS_LEVELV_NUM):
        # Yes, logger takes its '*args' as 'args'.
        self._log(PROGRESS_LEVELV_NUM, message, args, **kws) 
logging.Logger.progress = progress

class ConsoleFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    green = "\x1b[32;20m"
    reset = "\x1b[0m"
    format = " %(levelname)s: %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
        logging.PROGRESS: green + format + reset
    }
    
    def __init__(self):
        self.start_time = time.time()

    def format(self, record):

        elapsed_seconds = record.created - self.start_time
        #using timedelta here for convenient default formatting
        elapsed = datetime.timedelta(seconds = elapsed_seconds)

        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return f'{elapsed} {formatter.format(record)}'
    
class FileFormatter(logging.Formatter):

    format = "%(levelname)s: %(message)s"

    FORMATS = {
        logging.DEBUG: format,
        logging.INFO: format,
        logging.WARNING: format,
        logging.ERROR: format,
        logging.CRITICAL: format,
        logging.PROGRESS: format
    }

    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = record.created - self.start_time
        #using timedelta here for convenient default formatting
        elapsed = datetime.timedelta(seconds = elapsed_seconds)

        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return f'{elapsed} {formatter.format(record)}'
    
def get_log_name(log_folder: str) -> str:
    """get log name.if there is a log.txt, iterate integers log_0.txt, log_1txt, etc until a new log name is found.

    Parameters
    ----------
    log_folder : str
        log folder

    Returns
    -------
    log_name : str
        log name

    """
    log_name = os.path.join(log_folder, 'log.txt')
    i = 0
    while os.path.exists(log_name):
        log_name = os.path.join(log_folder, f'log_{i}.txt')
        i += 1
    return log_name

def init_logging(log_folder: str = None, log_level: int = logging.INFO):

    # create logger with 'spam_application'
    logger = logging.getLogger()
    # clear all ahndlers
    logger.handlers = []
    logger.setLevel(log_level)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(ConsoleFormatter())

    # add the handlers to the logger
    logger.addHandler(ch)

    if log_folder is not None:

        log_name = get_log_name(log_folder)
        # create file handler which logs even debug messages
        fh = logging.FileHandler(log_name)
        fh.setLevel(log_level)
        fh.setFormatter(FileFormatter())
        logger.addHandler(fh)