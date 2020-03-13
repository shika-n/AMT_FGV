from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QGridLayout, QWidget, QSplitter, QScrollBar, QAction, QFileDialog
from scipy.io import wavfile
from GUI.Graphics.Image import Image
from GUI.Graphics.AudioSpectrum import AudioSpectrum
from GUI.Graphics.AudioSpectrogram import AudioSpectrogram


class Window (QMainWindow):
    def __init__(self):
        super(Window, self).__init__()
        self.setWindowTitle('AudioToScore')

        # Variables
        view_width = 1024 * 1
        view_height = 128
        self.__spectrogram_view = AudioSpectrogram(view_width, 512)
        self.__audio_view = AudioSpectrum(view_width, view_height)
        self.test_widget = Image(300, 300)
        self.__main_splitter = QSplitter()

        self.__view_controller_layout = QGridLayout()
        self.__view_controller_layout_widget = QWidget()

        self.__audio_view_container = QWidget()
        self.__spectrogram_view_container = QWidget()

        self.__view_v_scroll_bar = QScrollBar()
        self.__view_h_scroll_bar = QScrollBar()

        # Menu Variables
        self.__main_menu = self.menuBar().addMenu('&File')
        self.__open_action = QAction('&Open')
        self.__exit_action = QAction('E&xit')

        self.__main_menu.addAction(self.__open_action)
        self.__main_menu.addSeparator()
        self.__main_menu.addAction(self.__exit_action)

        # Widgets properties
        self.__audio_view.setMinimumSize(640, view_height)
        self.__spectrogram_view.setMinimumSize(640, 256)

        self.__audio_view_container.setMinimumWidth(640)
        self.__audio_view_container.setFixedHeight(view_height)
        self.__audio_view_container.setStyleSheet('overflow: hidden;')
        self.__spectrogram_view_container.setMinimumWidth(640)
        self.__spectrogram_view_container.setMinimumHeight(256)
        self.__spectrogram_view_container.setStyleSheet('overflow: hidden;')

        self.__view_v_scroll_bar.setMaximum(self.__spectrogram_view.get_height())
        self.__view_v_scroll_bar.setPageStep(1024)
        self.__view_h_scroll_bar.setOrientation(Qt.Horizontal)
        self.__view_h_scroll_bar.setMaximum(view_width)
        self.__view_h_scroll_bar.setPageStep(1024)

        # Events
        self.__open_action.triggered.connect(self.__open_file)
        self.__exit_action.triggered.connect(self.exit_application)

        self.__view_v_scroll_bar.sliderMoved.connect(self.__view_v_scrolled)
        self.__view_h_scroll_bar.sliderMoved.connect(self.__view_h_scrolled)

        # Adding widgets
        self.__audio_view.setParent(self.__audio_view_container)
        self.__spectrogram_view.setParent(self.__spectrogram_view_container)

        self.__view_controller_layout.addWidget(self.__audio_view_container, 0, 1)
        self.__view_controller_layout.addWidget(self.__spectrogram_view_container, 1, 1)

        self.__view_controller_layout.addWidget(self.__view_v_scroll_bar, 1, 2)
        self.__view_controller_layout.addWidget(self.__view_h_scroll_bar, 2, 1)

        # Layouts
        self.__view_controller_layout_widget.setLayout(self.__view_controller_layout)

        self.__main_splitter.addWidget(self.__view_controller_layout_widget)
        self.__main_splitter.addWidget(self.test_widget)
        self.setCentralWidget(self.__main_splitter)

        self.statusBar().value = 'Ready'

        self.show()
        screen_rect = QApplication.desktop().screenGeometry()
        self.move((screen_rect.width() - self.width()) / 2,
                  (screen_rect.height() - self.height()) / 2)

    def __view_v_scrolled(self, value):
        normalized_value = value / self.__view_v_scroll_bar.maximum()
        self.__spectrogram_view.move(self.__spectrogram_view.x(), -normalized_value * (self.__spectrogram_view.get_height() - 200))

    def __view_h_scrolled(self, value):
        normalized_value = value / self.__view_h_scroll_bar.maximum()
        self.__audio_view.move(-normalized_value * (self.__audio_view.get_width() - 200), self.__audio_view.y())
        self.__spectrogram_view.move(-normalized_value * (self.__spectrogram_view.get_width() - 200), self.__spectrogram_view.y())

    def __open_file(self):
        file_path = QFileDialog.getOpenFileName(self, filter='WAV File (*.wav);;All Files (*.*)')[0]
        if file_path:
            sample_rate, data = wavfile.read(file_path)
            self.__audio_view.construct_from_data(sample_rate, data)
            self.__spectrogram_view.construct_from_data(sample_rate, data)

    def exit_application(self):
        self.close()
