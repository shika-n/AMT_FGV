from PyQt5.QtCore import pyqtSignal, pyqtSlot, QThread, QObject
import numpy as np
from GUI.Graphics.Image import Image
from GUI.LoadingWindow import LoadingWindow


class AudioSpectrum (Image):

    def __init__(self, width, height):
        super(AudioSpectrum, self).__init__(width, height)

        self.__sample_rate = 0
        self.__data = []

        self.__worker_thread = QThread()
        self.__loading_window = LoadingWindow()
        self.__spectrum_worker = SpectrumWorker(self._width, self._height)

    def construct_from_data(self, sample_rate, data):
        self.__sample_rate = sample_rate
        self.__data = data

        self.clear()
        self.__loading_window.show()

        self.__spectrum_worker.set_data(data)
        self.__spectrum_worker.moveToThread(self.__worker_thread)
        self.__spectrum_worker.signal_update_progress.connect(self.__loading_window.set_progress)
        self.__spectrum_worker.signal_update_status.connect(self.__loading_window.set_status)
        self.__spectrum_worker.signal_worker_done.connect(self.__worker_done)
        self.__spectrum_worker.signal_draw_line.connect(self.draw_line)
        self.__worker_thread.started.connect(self.__spectrum_worker.run)
        self.__worker_thread.start()

    def __worker_done(self):
        self.__loading_window.hide()
        self.update_pixmap()
        self.__worker_thread.quit()


class SpectrumWorker (QObject):
    signal_update_progress = pyqtSignal(float)
    signal_update_status = pyqtSignal(str)
    signal_draw_line = pyqtSignal(int, int, int, int)
    signal_worker_done = pyqtSignal()

    def __init__(self, width, height):
        super(SpectrumWorker, self).__init__()
        self.__width = width
        self.__height = height
        self.__data = []

    def set_data(self, data):
        self.__data = data

    def run(self):
        data_length = self.__data.shape[0]
        length_to_width_ratio = data_length / self.__width

        self.signal_update_status.emit('Resizing data...')
        resized_data = []
        for i in range(0, int(data_length / length_to_width_ratio)):
            amp = np.sum(self.__data[int(i * length_to_width_ratio):int((i + 1) * length_to_width_ratio)], axis=0)
            resized_data.append(amp)
            self.signal_update_progress.emit(i / int(data_length / length_to_width_ratio))

        max = np.max(resized_data)
        norm = (self.__height / 2) / max

        self.signal_update_status.emit('Drawing the spectrum...')
        last_data = resized_data[0]
        for x in range(1, self.__width):
            current_data = resized_data[x]
            self.signal_draw_line.emit(x - 1, int(last_data[0] * norm + self.__height / 2),
                                x, int(current_data[0] * norm + self.__height / 2))
            last_data = current_data
            self.signal_update_progress.emit(x / self.__width)

        self.signal_worker_done.emit()
