import numpy as np
from PyQt5.QtCore import QObject
from GUI.Graphics.Image import Image
from GUI.LoadingWindow import LoadingWindow

class AudioSpectrogram (Image):
    def __init__(self, width, height):
        super(AudioSpectrogram, self).__init__(width, height)

        self.__loading_window = LoadingWindow()

    def construct_from_data(self, data):
        print('Constructing spectrogram')


class SpectrogramWorker (QObject):
    def __init__(self):
        super(SpectrogramWorker, self).__init__()

    def run(self):
        print('Thread running..')