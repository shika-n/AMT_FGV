import time

class Timer:
    def __init__(self):
        self.start = self.__get_time()

    def tic(self):
        self.start = self.__get_time()

    def toc(self):
        return (self.__get_time() - self.start) / 1000000000

    def __get_time(self):
        return time.perf_counter_ns()
