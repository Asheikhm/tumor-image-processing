from PyQt5 import QtWidgets as QtW
from PyQt5 import QtCore as QtC

class main_window(QtW.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Snake Game")
        self.application = qt.Qt.QApplication