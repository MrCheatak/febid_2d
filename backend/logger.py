import logging
import re


def colored(text, t=None, b=None):
    """
    Color text based on the given scale.
    :param text: text to color
    :param t: text color as RGB tuple
    :param b: background color as RGB tuple
    :return:
    """
    if t is None:
        t = (0,0,0)
    if b is None:
        b = (255, 255, 255)
    return f"\033[48;2;{b[0]};{b[1]};{b[2]}m\033[38;2;{t[0]};{t[1]};{t[2]}m{text}\033[0m"


def green_red(val):
    """
    Define a green-red color scale.
    :param val: value between 0 and 1
    :return: tuple of RGB values
    """
    i = int(85 * val)
    if i < 0:
        i = 0
    if i > 85:
        i = 85
    r = i
    g = 85 - i
    return (120+r, 120+g, 120)


class RemoveColorCodes(logging.Filter):
    def filter(self, record):
        record.msg = re.sub(r'\x1b\[[0-9;]*m', '', record.msg)
        return True


class Logger:
    """
    Logger class with colored output.
    """
    def __init__(self, name):
        self.name = name
        self.logger = self.get_logger()

    def colored(self, text, t=None, b=None):
        """
        Color text and background with the provided colors.
        :param text: text to color
        :param t: text color as RGB tuple
        :param b: background color as RGB tuple
        :return:
        """
        return colored(text, t, b)

    def green_red(self, val):
        """
        Get an RGB color of the value on the green-red colormap.
        :param val: value between 0 and 1
        :return: tuple of RGB values
        """
        return green_red(val)

    def green_red_text(self, text, t=None, b=None):
        """
        Color text and background with the green-red colormap.
        :param text: string to color
        :param t: text color, a value between 0 and 1 on the green-red colormap
        :param b: background color, a value between 0 and 1 on the green-red colormap
        :return: a string with color codes for console output
        """
        if t is not None:
            t_val = self.green_red(t)
        else:
            t_val = None
        if b is not None:
            b_val = self.green_red(b)
        else:
            b_val = None
        return self.colored(text, t_val, b_val)

    def get_logger(self):
        """
        Setup the logger
        :return:
        """
        fname = f'{self.name}.log'
        logger = logging.getLogger(fname)
        logger.setLevel(logging.DEBUG)

        fmt_console = '%(asctime)s [%(levelname)s] %(color_code_opening)s%(message)s%(color_code_closing)s'
        formatter_console = logging.Formatter(fmt_console)
        fmt_file = '%(asctime)s [%(levelname)s] %(message)s'
        formatter_file = logging.Formatter(fmt_file)

        file_handler = logging.FileHandler(fname)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter_file)
        # file_handler.addFilter(RemoveColorCodes())

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter_console)
        print(stream_handler.filters)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        return logger

    def set_file_output(self, flag):
        pass

    def log_message(self, message, level, text_color=None, background_color=None):
        """
        Log a message with the given level and color(console output).
        :param message: text to log
        :param level: logging level
        :param text_color: a value between 0 and 1 on the green-red colormap
        :param background_color: a value between 0 and 1 on the green-red colormap
        :return:
        """
        message_colored = self.green_red_text(message, t=text_color, b=background_color)
        c_code_opening = message_colored[0:message_colored.rindex(message)]
        c_code_closing = '\033[0m'
        self.logger.log(level, message, extra={'color_code_opening': c_code_opening,
                                               'color_code_closing': c_code_closing})

    def debug(self, message, text_color=None, background_color=None):
        """
        Log a debug message with the given color.
        :param message: text to log
        :param text_color: a value between 0 and 1 on the green-red colormap
        :param background_color: a value between 0 and 1 on the green-red colormap
        :return:
        """
        self.log_message(message, logging.DEBUG, text_color, background_color)

    def info(self, message, text_color=None, background_color=None):
        """
        Log an info message with the given color.
        :param message: text to log
        :param text_color: a value between 0 and 1 on the green-red colormap
        :param background_color: a value between 0 and 1 on the green-red colormap
        :return:
        """
        self.log_message(message, logging.INFO, text_color, background_color)

    def warning(self, message, text_color=None, background_color=None):
        """
        Log a warning message with the given color.
        :param message: text to log
        :param text_color: a value between 0 and 1 on the green-red colormap
        :param background_color: a value between 0 and 1 on the green-red colormap
        :return:
        """
        self.log_message(message, logging.WARNING, text_color, background_color)

    def error(self, message, text_color=None, background_color=None):
        """
        Log an error message with the given color.
        :param message: text to log
        :param text_color: a value between 0 and 1 on the green-red colormap
        :param background_color: a value between 0 and 1 on the green-red colormap
        :return:
        """
        self.log_message(message, logging.ERROR, text_color, background_color)

    def critical(self, message, text_color=None, background_color=None):
        """
        Log a critical message with the given color.
        :param message: text to log
        :param text_color: a value between 0 and 1 on the green-red colormap
        :param background_color: a value between 0 and 1 on the green-red colormap
        :return:
        """
        self.log_message(message, logging.CRITICAL, text_color, background_color)


if __name__ == '__main__':
    # logger = get_logger('test')
    logger = Logger('test')
    logger.info('This is an info message test', background_color=0.8)
    logger.debug('This is a debug message test', text_color=0.5)
    logger.error('This is an error message test', text_color=0.1, background_color=0.9)
    logger.warning('This is a warning message test', text_color=0.9, background_color=0.1)
    logger.critical('This is a critical message test', text_color=0.3, background_color=0.7)

