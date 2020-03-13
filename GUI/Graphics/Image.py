import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor


class Image (QLabel):
    def __init__(self, width, height):
        super(Image, self).__init__()
        self._width = width
        self._height = height
        self.__image = QImage(width, height, QImage.Format_RGB32)
        self.__painter = QPainter(self.__image)

        self.clear_image()

        self.draw_line(0, 0, width, height)
        self.draw_line(0, height, width, 0)

        self.update_pixmap()

    def resize_component(self, width, height):
        self._width = width
        self._height = height
        self.__image = QImage(width, height, QImage.Format_RGB32)

        self.update_pixmap()

    def update_pixmap(self):
        self.setPixmap(QPixmap.fromImage(self.__image))

    def clear_image(self):
        self.__image.fill(0)
        self.update_pixmap()

    def draw_line(self, x0, y0, x1, y1):
        self.__painter.setPen(QPen(Qt.white, 1))
        self.__painter.drawLine(x0, y0, x1, y1)

    def draw_heatmap(self, x, data):
        data_length = data.shape[0]
        for y in range(0, self._height):
            value = int(data[int(data_length * y / self._height)])
            self.__image.setPixel(x, y, value)

    # for testing purposes
    def fill_gradient(self):
        for y in range(0, self._height):
            for x in range(0, self._width):
                self.__image.setPixel(x, y, (x << 16) | (y << 8))

        self.update_pixmap()

    def get_width(self):
        return self._width

    def get_height(self):
        return self._height
