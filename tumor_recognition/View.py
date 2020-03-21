from PyQt5 import QtWidgets as QtW
from PyQt5 import QtCore as QtC
from PyQt5 import QtGui as QtG

class applicationMenu(QtW.QWidget):

    def __init__(self, parent):
        super().__init__()
        self.__parent = parent
        self.__label = QtW.QLabel('Number of images :')
        self.__nb_images = QtW.QLineEdit()
        self.__nb_images.setMaximumWidth(150)

        self.__prepare = QtW.QPushButton('Prepare')
        self.__learn = QtW.QPushButton('Learn')

        self.__select = QtW.QPushButton('Select')
        self.__path = QtW.QLineEdit(' ')
        self.__path.setReadOnly(True)

    def set_layouts(self):
        self.layoutHv1 = QtW.QHBoxLayout()
        self.layoutHv1.addWidget(self.__label)
        self.layoutHv1.addWidget(self.__nb_images)

        self.layoutHv2 = QtW.QHBoxLayout()
        self.layoutHv2.addWidget(self.__prepare)
        self.layoutHv2.addWidget(self.__learn)

        self.layoutHv3 = QtW.QHBoxLayout()
        self.layoutHv3.addWidget(self.__select)
        self.layoutHv3.addWidget(self.__path)

        self.main_layout = QtW.QVBoxLayout()
        self.main_layout.addLayout(self.layoutHv1)
        self.main_layout.addLayout(self.layoutHv2)
        self.main_layout.addLayout(self.layoutHv3)
        self.__select.clicked.connect(self.open_file_dialog)        
        self.setLayout(self.main_layout)
    
    def open_file_dialog(self):
        options = QtW.QFileDialog.Options()
        filename, _ = QtW.QFileDialog.getOpenFileName(self,"Choose your image", "","All Files (*);;Python Files (*.py)", options=options)

        if filename:
            self.set_path(filename)
            self.__parent.update_image(filename)

    def set_path(self, filename):
        self.__path.setText(filename)

    @property
    def path(self):
        return self.__path

class applicationCanvas(QtW.QWidget):

    def __init__(self, parent):
        super().__init__()
        self.__parent = parent

    def set_canvas(self):
        self.__image = QtG.QPixmap()
        self.__label = QtW.QLabel()
        self.__label.setMinimumWidth(400)
        self.__label.setMaximumHeight(700)
        self.__label.setPixmap(self.__image)

        self.layout = QtW.QGridLayout()
        self.layout.addWidget(self.__label,1,1)
        self.setLayout(self.layout)

    def update_image(self,file_path):
        self.__image.load(file_path)
        self.__label.setPixmap(self.__image)

class mainWindow(QtW.QWidget):

    def __init__(self, width = 1000, height = 800):
        self.__application = QtW.QApplication([])
        super().__init__()
        self.title = "Tumor recognition"
        self.setWindowTitle("Snake Game")
        self.__width = width
        self.__height = height
        self.__menu = applicationMenu(self)
        self.__canvas = applicationCanvas(self)
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(0, 0, self.__width, self.__height)
        self.__layout = QtW.QHBoxLayout()
        self.__layout.addWidget(self.__menu)
        self.__layout.addWidget(self.__canvas)
        self.__menu.set_layouts()
        self.__canvas.set_canvas()
        self.setLayout(self.__layout)
        self.show()
        self.__application.exec_()

    def update_image(self, image_path):
        self.__canvas.update_image(image_path)


if __name__ == '__main__':
    ex = mainWindow()