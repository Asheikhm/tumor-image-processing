import Controller
import View
import Model
import numpy
from PIL import Image

def build_application():
    model = Model.TumorRecognitionBrain()
    view = View.MainWindow()
    controller = Controller.TumorRecognitionApplication(view, model)

    prepare_button = View.PrepareButtonView('Prepare', controller)
    select_button = View.SelectButtonView('Select', controller)
    learn_button = View.LearnButtonView('Learn', controller)
    
    view.menu().initialize_buttons(select_button, prepare_button, learn_button)
    view.initUI()

def gaussian( x, y, avg_x, avg_y, sig_x, sig_y):
    value = (1 / (numpy.sqrt(2 * numpy.pi) * sig_x * sig_y)) * numpy.exp(-(((x - avg_x) ** 2) / (2 * sig_x) + ((y - avg_y) ** 2) / (2 * sig_y)))
    return value

if __name__ == '__main__':
    build_application()
    #kernel = numpy.zeros((3,3))
    #sig_x = (3 - 1) / 5
    #sig_y = sig_x

    #avg_x = ((3 - 1) / 2)
    #avg_y = avg_x

    #populate kernel
    #rows, cols = kernel.shape
    #for i in range(rows):
        #x = numpy.ones((cols)) * i 
        #y = numpy.arange(cols)
        #kernel[i,:] = gaussian(x, y, avg_x, avg_y, sig_x, sig_y)

    #normalize kernel
    #sum_values = numpy.sum(kernel)
    #kernel = kernel / sum_values
    #c = numpy.arange(20).reshape(4,5)
    #image_padded = numpy.pad(c, pad_width = 1, mode = 'constant', constant_values = 0)
    #submatrix = numpy.zeros((3, 3))
    #image_output1 = numpy.zeros((4, 5))

    #Create array that contains submatrices
    #for indexes, value in numpy.ndenumerate(c):
        #submatrix = numpy.zeros((kernel_size, kernel_size))
        #submatrix = image_padded[indexes[0] : indexes[0] + 3, indexes[1] : indexes[1] + 3]
        #image_output1[indexes[0]][indexes[1]] = numpy.multiply(submatrix, kernel).sum()
    #print(image_output1)

    #image_output2 = numpy.zeros((4, 5))
    #rows_kernel, cols_kernel = kernel.shape
    #rows1, cols1 = c.shape
    #submatrix2 = numpy.zeros((3, 3))

    #Horizontal convolution
    #for i in range(rows1):
        #print(c[i])
        #submatrix2 = image_padded[indexes[0] : indexes[0] + 3]
        #image_output2[indexes[0]][indexes[1]] = numpy.multiply(rows_kernel[0], submatrix2).sum()

        #pour chaque pixel, multiplie par row_kernel et additionne le tout
        #[0] = ([1,2,3,4,5] * [4,5,6,7,8])
        #pass