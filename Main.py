from PyQt5.QtWidgets import QApplication
from GUI.Window import Window


def main():
    app = QApplication([])
    wind = Window()
    app.exec()


if __name__ == '__main__':
    main()