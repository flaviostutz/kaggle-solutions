import logging

def logger0():
    global _console_logger
    if(not _console_logger):
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        logger.addHandler(sh)
        _console_logger = True
    return logging.getLogger()

def setup_file_logger(log_file):
    global _file_logger
    global logger
    if(not _file_logger):
        hdlr = logging.FileHandler(log_file)
        hdlr.setLevel(logging.DEBUG)
        hdlr.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        logger.addHandler(hdlr) 
        _file_logger = True

_console_logger = False
_file_logger = False
logger = logger0()

