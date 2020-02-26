from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QWidget, QProgressBar, QLabel, QVBoxLayout


class LoadingWindow (QWidget):

    def __init__(self):
        super(LoadingWindow, self).__init__()
        self.setWindowTitle('Loading')

        self.setLayout(QVBoxLayout())
        self.__progress_bar = QProgressBar()
        self.__status_label = QLabel()

        self.__progress_bar.setRange(0, 1000)

        self.layout().addWidget(self.__progress_bar)
        self.layout().addWidget(self.__status_label)

    @pyqtSlot(float)
    def set_progress(self, value):
        self.__progress_bar.setValue(value * 1000)

    @pyqtSlot(str)
    def set_status(self, status):
        self.__status_label.setText(status)


