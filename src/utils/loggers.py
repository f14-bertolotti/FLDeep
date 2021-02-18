import logging

class DefaultLogger:

    @staticmethod
    def get_logger(name, file=None, stream=None, formatter=None):
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        if file:
            fh = logging.FileHandler(file, mode="a")
            fh.setFormatter(logging.Formatter(formatter))
            logger.addHandler(fh)
        if stream:
            fh = logging.StreamHandler(stream)
            fh.setFormatter(logging.Formatter(formatter))
            logger.addHandler(fh)
        return logger


    @staticmethod
    def get_fake_logger():
        class Fake:
            def __init__(self): pass
            def   info(self,x): pass
        return Fake()
