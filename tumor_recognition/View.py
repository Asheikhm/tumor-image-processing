from PyQt5 import QtWidgets as QtW
from PyQt5 import QtCore as QtC
from PyQt5 import QtGui as QtG
import time


class PrepareButtonView(QtW.QPushButton):
    def __init__(self, name, controller):
        super(QtW.QPushButton, self).__init__(name)
        self.__controller = controller

    def prepare_images(self):
        self.__controller.create_tumors()

class SelectButtonView(QtW.QPushButton):
    def __init__(self, name, controller):
        super(QtW.QPushButton, self).__init__(name)
        self.__controller = controller

class LearnButtonView(QtW.QPushButton):
    def __init__(self, name, controller):
        super(QtW.QPushButton, self).__init__(name)
        self.__controller = controller
    
    def learning(self):
        self.__controller.learn()

class ApplicationMenu(QtW.QWidget):
    def __init__(self, parent):
        super().__init__()
        self.__parent = parent
        self.__label = QtW.QLabel('Number of images (minimum 500):')
        self.__nb_images = QtW.QLineEdit()
        self.__nb_images.setMaximumWidth(150)

        self.__prepare = None
        self.__learn = None

        self.__select = None
        self.__path = QtW.QLineEdit(' ')
        self.__path.setReadOnly(True)

        self.__label_SaltNPepper = QtW.QLabel('Salt and Pepper noise')
        self.__checkSaltNPepper = QtW.QCheckBox()
        self.__label_Gaussian = QtW.QLabel('Gaussian noise')
        self.__checkGaussian = QtW.QCheckBox()

        self.bg = QtW.QButtonGroup()
        self.bg.addButton(self.__checkSaltNPepper,1)
        self.bg.addButton(self.__checkGaussian,2)

        #Apply Gaussian noise by default
        self.__checkGaussian.setChecked(True)

    def set_layouts(self):
        self.layoutHv1 = QtW.QHBoxLayout()
        self.layoutHv1.addWidget(self.__label)
        self.layoutHv1.addWidget(self.__nb_images)

        self.layoutHv2 = QtW.QHBoxLayout()
        self.layoutHv2.addWidget(self.__prepare)
        self.layoutHv2.addWidget(self.__learn)

        self.layoutHv3 = QtW.QGridLayout()
        self.layoutHv3.addWidget(self.__label_SaltNPepper, 1, 1)
        self.layoutHv3.addWidget(self.__checkSaltNPepper, 1, 2)
        self.layoutHv3.addWidget(self.__label_Gaussian, 2, 1)
        self.layoutHv3.addWidget(self.__checkGaussian, 2, 2)

        self.layoutHv4 = QtW.QHBoxLayout()
        self.layoutHv4.addWidget(self.__select)
        self.layoutHv4.addWidget(self.__path)

        self.main_layout = QtW.QVBoxLayout()
        self.main_layout.addLayout(self.layoutHv1)
        self.main_layout.addLayout(self.layoutHv2)
        self.main_layout.addLayout(self.layoutHv3)
        self.main_layout.addLayout(self.layoutHv4)

        self.setLayout(self.main_layout)
    
    def connect(self):
        self.__select.clicked.connect(self.open_file_dialog)
        self.__prepare.clicked.connect(self.__prepare.prepare_images)
        self.__learn.clicked.connect(self.__learn.learning)

    def transfer_informations(self):
        if self.__nb_images.text() != '':
            self.parent.send_nb_images(self.__nb_images.text())

    def open_file_dialog(self):
        options = QtW.QFileDialog.Options()
        filename, _ = QtW.QFileDialog.getOpenFileName(self,'Choose your image', '','All Files (*);;Python Files (*.py)', options=options)

        if filename:
            self.set_path(filename)
            self.__parent.update_image(filename)

    def set_path(self, filename):
        self.__path.setText(filename)

    @property
    def path(self):
        return self.__path

    def initialize_buttons(self, select_button, prepare_button, learn_button):
        self.__select = select_button
        self.__prepare = prepare_button
        self.__learn = learn_button

class ApplicationCanvas(QtW.QWidget):
    def __init__(self, parent):
        super().__init__()
        self.__parent = parent

    def set_canvas(self):
        self.__image = QtG.QPixmap()
        self.setStyleSheet('background-color : black;')
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

class MainWindow(QtW.QWidget):
    def __init__(self, width = 1000, height = 800):
        self.__application = QtW.QApplication([])
        super().__init__()
        self.title = "Tumor recognition"
        self.__width = width
        self.__height = height
        self.__menu = ApplicationMenu(self)
        self.__canvas = ApplicationCanvas(self)

    def initUI(self):

        self.setWindowTitle(self.title)
        self.setGeometry(0, 0, self.__width, self.__height)
        self.__layout = QtW.QGridLayout()
        self.__layout.addWidget(self.__menu, 1, 1, 1, 1)
        self.__layout.addWidget(self.__canvas, 1, 2, 1, 4)
        self.__menu.set_layouts()
        self.__menu.connect()
        self.__canvas.set_canvas()
        self.setLayout(self.__layout)
        self.show()
        self.__application.exec_()

    def update_image(self, image_path):
        self.__canvas.update_image(image_path)
    
    def menu(self):
        return self.__menu
        
    def selected_noise_Strategy(self):
        return self.__menu.bg.checkedId()