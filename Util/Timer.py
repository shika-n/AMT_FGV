import time

class Timer:
    def __init__(self, exit_msg=''):
        self.start = self.__get_time()
        self.exit_msg = exit_msg

    def tic(self):
        self.start = self.__get_time()

    def toc(self):
        return (self.__get_time() - self.start) / 1000000000

    def __get_time(self):
        return time.perf_counter_ns()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(self.exit_msg, 'took', self.toc(), 'second(s)')