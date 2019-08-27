import time
import random

def make_name_hash(name='', file_ending='.pt'):
    year, month, day, hour, minute, second = time.localtime()[:6]
    ascii_letters_digits = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    random_hash = ''.join(random.choices(ascii_letters_digits, k=20))
    path = f"{name}_{year}-{month}-{day}_{hour}-{minute}-{second}_{random_hash}{file_ending}"
    return path

class TimeLogger:
    def __init__(self, start=None):
        self.start = self.time() if start is None else start
        self.prev = self.start

    @staticmethod
    def time():
        return time.time()

    def diff(self):
        prev, self.prev = (self.prev, self.time())
        return self.prev - self.start, self.prev - prev

    @staticmethod
    def _hms_from_sec(sec):
        """Hours, minutes, seconds."""
        m, s = divmod(sec, 60)
        h, m = divmod(m, 60)
        return h, m, s

    @staticmethod
    def _hms_str(h, m, s, shorten=True):
        """Hours, minutes, seconds."""
        hs = f"{int(h)}h:"
        ms = f"{int(m)}m:"
        ss = f"{int(s)}s"
        if shorten:
            if h == 0:
                hs = ''
                if m == 0:
                    ms = ''
        return f"{hs}{ms}{ss}"
        # return f"{int(h)}h:{int(m)}m:{int(s)}s"

    def hms_diff(self, shorten=True):
        diff_start, diff_prev = self.diff()
        hms_start = self._hms_from_sec(diff_start)
        hms_prev = self._hms_from_sec(diff_prev)
        return self._hms_str(*hms_start, shorten), self._hms_str(*hms_prev, shorten)