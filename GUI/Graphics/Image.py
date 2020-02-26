from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QImage, QPixmap


class Image (QLabel):
    def __init__(self, width, height):
        super(Image, self).__init__()
        self._width = width
        self._height = height
        self._image = QImage(width, height, QImage.Format_RGB32)

        self.clear()

        self.draw_line(0, 0, width, height)
        self.draw_line(0, height, width, 0)

        self.update_pixmap()

    def resize_component(self, width, height):
        self._width = width
        self._height = height
        self._image = QImage(width, height, QImage.Format_RGB32)

        self.update_pixmap()

    def update_pixmap(self):
        self.setPixmap(QPixmap.fromImage(self._image))

    def clear(self):
        self._image.fill(0)
        self.update_pixmap()

    def draw_line(self, x0, y0, x1, y1):
        x_diff = x1 - x0
        y_diff = y1 - y0

        if x_diff == 0:
            for y in range(0, y_diff):
                self._image.setPixel(x0, y0 + y, 0xffffff)
            for y in range(0, y_diff, -1):
                self._image.setPixel(x0, y0 + y, 0xffffff)
        elif y_diff == 0:
            for x in range(0, x_diff):
                self._image.setPixel(x0 + x, y0, 0xffffff)
            for x in range(0, x_diff, -1):
                self._image.setPixel(x0 + x, y0, 0xffffff)
        else:
            x_length = abs(x_diff)
            y_length = abs(y_diff)
            x_sign = int(x_diff / x_length)
            y_sign = int(y_diff / y_length)

            if x_length > y_length:
                for x in range(0, x_length):
                    self._image.setPixel(x0 + x * x_sign, int(y0 + y_diff * (x / x_length)), 0xffffff)
            else:
                for y in range(0, y_length):
                    self._image.setPixel(int(x0 + x_diff * (y / y_length)), y0 + y * y_sign, 0xffffff)

        self._image.setPixel(x1, y1, 0xffffff)

    def fill_gradient(self):
        for y in range(0, self._height):
            for x in range(0, self._width):
                self._image.setPixel(x, y, (x << 16) | (y << 8))

        self.update_pixmap()

    def get_width(self):
        return self._width

    def get_height(self):
        return self._height
