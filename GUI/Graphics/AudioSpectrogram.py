import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, QThread
from GUI.Graphics.Image import Image
from GUI.LoadingWindow import LoadingWindow
from Util.Timer import Timer


class AudioSpectrogram (Image):
    def __init__(self, width, height):
        super(AudioSpectrogram, self).__init__(width, height)

        self.__sample_rate = 0
        self.__data = []

        self.__worker_thread = QThread()
        self.__loading_window = LoadingWindow()
        self.__spectrogram_worker = SpectrogramWorker(self._width, self._height)

        self.__spectrogram_worker.moveToThread(self.__worker_thread)
        self.__spectrogram_worker.signal_update_progress.connect(self.__loading_window.set_progress)
        self.__spectrogram_worker.signal_update_status.connect(self.__loading_window.set_status)
        self.__spectrogram_worker.signal_draw_heatmap.connect(self.draw_heatmap)
        self.__spectrogram_worker.signal_worker_done.connect(self.__worker_done)

        self.__worker_thread.started.connect(self.__spectrogram_worker.run)

    def construct_from_data(self, sample_rate, data):
        self.__sample_rate = sample_rate
        self.__data = data

        self.clear_image()
        self.__loading_window.show()

        self.__spectrogram_worker.set_data(self.__sample_rate, data)

        self.__worker_thread.start()

    def __worker_done(self):
        self.__loading_window.hide()
        self.update_pixmap()
        self.__worker_thread.quit()


class SpectrogramWorker (QObject):
    signal_update_progress = pyqtSignal(float)
    signal_update_status = pyqtSignal(str)
    signal_draw_heatmap = pyqtSignal(int, np.ndarray)
    signal_worker_done = pyqtSignal()

    def __init__(self, width, height):
        super(SpectrogramWorker, self).__init__()
        self.__width = width
        self.__height = height
        self.__data = []
        self.__sample_rate = 0;

    def set_data(self, sample_rate, data):
        self.__sample_rate = sample_rate
        self.__data = data

    def run(self):
        timer = Timer()

        # __data is in raw form, [[L, R], [L, R], ...]
        data_length = self.__data.shape[0]
        length_to_width_ratio = data_length / self.__width

        half_window_length = int(self.__sample_rate * 0.1)
        # Take only the first column containing L samples
        mono_data = self.__data[:, 0]

        self.signal_update_status.emit('Generating audio spectrogram')

        for i in range(0, self.__width):
            data_center_index = int(i * length_to_width_ratio)
            start_window = data_center_index - half_window_length
            end_window = data_center_index + half_window_length
            if start_window < 0:
                start_window = 0
            if end_window > mono_data.shape[0]:
                end_window = mono_data.shape[0]

            data_window = mono_data[start_window:end_window]
            window = np.hanning(data_window.shape[0])
            windowed_data = np.multiply(data_window, window)

            fft_result = np.abs(np.fft.rfft(windowed_data))
            #fft_result = np.sqrt(np.add(np.multiply(fft_result.real, fft_result.real), np.multiply(fft_result.imag, fft_result.imag)))

            self.signal_draw_heatmap.emit(i, fft_result.real);
            self.signal_update_progress.emit(i / self.__width)

        self.signal_worker_done.emit()

        print("AudioSpectrogram worker took", timer.toc(), "second(s)!")