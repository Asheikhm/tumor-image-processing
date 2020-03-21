import math
import numpy
import random
import time
from PIL import Image

class tumorParameters:
    def __init__(self):
        self.__avg_x = 0
        self.__avg_y = 0
        self.__sig_x = 0
        self.__sig_y = 0
        self.__theta = 0
        self.__amplitude = 0

    @property   
    def avg_x(self):
        return self.__avg_x

    @property
    def avg_y(self):
        return self.__avg_y

    @property
    def sig_x(self):
        return self.__sig_x

    @property
    def sig_y(self):
        return self.__avg_y

    @property
    def theta(self):
        return self.__theta

    @property
    def amplitude(self):
        return self.__amplitude

    @avg_x.setter
    def avg_x(self, x):
        self.__avg_x = x

    @avg_y.setter
    def avg_y(self, y):
        self.__avg_y = y

    @sig_x.setter
    def sig_x(self, x):
        self.__sig_x = x
  
    @sig_y.setter
    def sig_y(self, y):
        self.__avg_y = y

    @theta.setter
    def theta(self, theta):
        self.__theta = theta

    @amplitude.setter
    def amplitude(self, amplitude):
        self.__amplitude = amplitude

class tumorImage:
    def __init__(self, tumor_parameters, width_image = 500, height_image = 500):
        self.__tumor_parameters = tumor_parameters
        self.__width = width_image
        self.__height = height_image
        self.__pixels = numpy.zeros((width_image,height_image))
        self.noise = 0
    
    def get_pixels(self):
        return self.__pixels

    def get_pixel(self, x , y):
        return self.__pixels[x][y]
    
    def set_pixels (self, pixels):
        self.__pixels = pixels

    def get_width(self):
        return self.__width
    
    def get_height(self):
        return self.__height

class imageGeneratorEngine:
    def __init__(self, img_nb):
        self.__image_name = img_nb
        self.__tumor_parameters = tumorParameters()
        self.__image = tumorImage(self.__tumor_parameters)
        self.__coeff_a = 0
        self.__coeff_b = 0
        self.__coeff_c = 0

    def create_image(self):
        m = self.__image.get_pixels()
        rows, cols = m.shape
        for i in range(rows):
            x = numpy.ones((cols)) * i 
            y = numpy.arange(cols)
            m[i,:] = self.gaussian_function(x, y)
    
    def set_parameters(self):
        self.__tumor_parameters.avg_x = 400
        self.__tumor_parameters.avg_y = 250
        self.__tumor_parameters.sig_x = 30
        self.__tumor_parameters.sig_y = 20
        self.__tumor_parameters.theta = random.randrange(10,20)
        self.__tumor_parameters.amplitude = random.randrange(10,21)
        self.__set_gaussian_coefficients()

    def gaussian_function(self, x, y):
        f = self.__tumor_parameters.amplitude*(numpy.exp(-(self.__coeff_a*((x - self.__tumor_parameters.avg_x)**2) + 2*self.__coeff_b*(x - self.__tumor_parameters.avg_x)*(y - self.__tumor_parameters.avg_y) + self.__coeff_c*((y - self.__tumor_parameters.avg_y)**2)))) + self.__image.noise
        return f

    def __set_gaussian_coefficients(self):
        self.__coeff_a = (math.cos(self.__tumor_parameters.theta)**2)/(2*(self.__tumor_parameters.sig_x**2)) + (math.sin(self.__tumor_parameters.theta)**2) / (2*(self.__tumor_parameters.sig_y**2))
       
        self.__coeff_b = (math.sin(2*self.__tumor_parameters.theta)) / (4*(self.__tumor_parameters.sig_x**2)) + (math.sin(2*self.__tumor_parameters.theta)) / (4*(self.__tumor_parameters.sig_y**2))
       
        self.__coeff_c =  (math.sin(self.__tumor_parameters.theta)**2) / (2*(self.__tumor_parameters.sig_x**2)) + (math.cos(self.__tumor_parameters.theta)**2) / (2*(self.__tumor_parameters.sig_y**2))

    def convert_to_png_file(self):
        rescaled = (255.0 / self.__image.get_pixels().max() * (self.__image.get_pixels() - self.__image.get_pixels().min())).astype(numpy.uint8)

        im = Image.fromarray(rescaled)
        name = str(self.__image_name) +'.png'
        im.save(name)

class Population:
    def __init__(self, nb_images):
        self.__images = []
        self.__populate(nb_images)
    
    def creates_images(self):
        for image_generator in self.__images:
            image_generator.set_parameters()
            image_generator.create_image()
            image_generator.convert_to_png_file()

    def __populate(self, nb_images):
        for i in range(nb_images):
            image = imageGeneratorEngine(i)
            self.__images.append(image)

class tumorRecognitionBrain:
    def __init__(self):
        self.__space_memory = 50
        self.__population = Population(self.__space_memory)
    
    def create_tumors(self):
        self.__population.creates_images()

if __name__ == '__main__':
    tm = tumorRecognitionBrain()
    time1 = time.time()
    tm.create_tumors()
    print("--- %s seconds ---" % (time.time() - time1))