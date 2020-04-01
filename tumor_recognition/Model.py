import math
import numpy
import random
import time
import os
from PIL import Image

class TumorParameters:
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
        return self.__sig_y

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
        self.__sig_y = y

    @theta.setter
    def theta(self, theta):
        self.__theta = theta

    @amplitude.setter
    def amplitude(self, amplitude):
        self.__amplitude = amplitude

class TumorImage:
    def __init__(self, tumor_parameters, width_image = 10, height_image = 10):
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

class ImageGeneratorEngine:
    def __init__(self, img_nb):
        self.__image_name = img_nb + 1
        self.__tumor_parameters = TumorParameters()
        self.__image =TumorImage(self.__tumor_parameters)
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
        self.__tumor_parameters.avg_x = random.randrange(100,401)
        self.__tumor_parameters.avg_y = random.randrange(100,401)
        self.__tumor_parameters.sig_x = random.randrange(20,31)
        self.__tumor_parameters.sig_y = random.randrange(10,21)
        self.__tumor_parameters.theta = random.randrange(50,55)
        self.__tumor_parameters.amplitude = random.randrange(10,21)
        self.__set_gaussian_coefficients()

    def gaussian_function(self, x, y):
        f = self.__tumor_parameters.amplitude * (numpy.exp(-(self.__coeff_a*((x - self.__tumor_parameters.avg_x)**2) + 2*self.__coeff_b*(x - self.__tumor_parameters.avg_x)*(y - self.__tumor_parameters.avg_y) + self.__coeff_c*((y - self.__tumor_parameters.avg_y)**2))))
        return f

    def __set_gaussian_coefficients(self):
        self.__coeff_a = (math.cos(self.__tumor_parameters.theta)**2) / (2*(self.__tumor_parameters.sig_x**2)) + (math.sin(self.__tumor_parameters.theta)**2) / (2*(self.__tumor_parameters.sig_y**2))
       
        self.__coeff_b = (math.sin(2*self.__tumor_parameters.theta)) / (4*(self.__tumor_parameters.sig_x**2)) + (math.sin(2*self.__tumor_parameters.theta)) / (4*(self.__tumor_parameters.sig_y**2))
       
        self.__coeff_c = (math.sin(self.__tumor_parameters.theta)**2) / (2*(self.__tumor_parameters.sig_x**2)) + (math.cos(self.__tumor_parameters.theta)**2) / (2*(self.__tumor_parameters.sig_y**2))

    def convert_to_png_file(self):
        rescaled = (255.0 / self.__image.get_pixels().max() * (self.__image.get_pixels() - self.__image.get_pixels().min())).astype(numpy.uint8)
        im = Image.fromarray(rescaled)
        name = 'tumorImage_'+ str(self.__image_name)+'.png'
        currentDirectory = os.getcwd() + '/Batch_1/' + name
        im.save(currentDirectory,'PNG')

class Population:
    def __init__(self, nb_images):
        self.__image_generators = []
        self.__populate(nb_images)
    
    def creates_images(self):
        #Creates folder if it doesn't exist
        if not os.path.exists('Batch_1'):
            os.makedirs('Batch_1')

        for image_generator in self.__image_generators:
            image_generator.set_parameters()
            image_generator.create_image()
            image_generator.convert_to_png_file()

    def __populate(self, nb_images):
        for i in range(nb_images):
            image = ImageGeneratorEngine(i)
            self.__image_generators.append(image)

class FeatureFinder:
    def __init__(self):
        pass

class FeatureExtractionProcess:
    def __init__(self):
        self.images_features = FeatureFinder()

class LearningProcess:
    def __init__(self):
        self.__feature_extraction_process = FeatureExtractionProcess()
        self.__images_parameters = []

    def __extract_feature(self, image, kernel):
        self.__blur_image(image, kernel)

    def generate_kernel(self, size):
        kernel = numpy.zeros((size,size))
        sig_x = (size - 1) / 5
        sig_y = sig_x

        avg_x = ((size - 1) / 2)
        avg_y = avg_x

        #populate kernel
        rows, cols = kernel.shape
        for i in range(rows):
            x = numpy.ones((cols)) * i 
            y = numpy.arange(cols)
            kernel[i,:] = self.gaussian(x, y, avg_x, avg_y, sig_x, sig_y)

        #normalize kernel
        sum_values = numpy.sum(kernel)
        kernel = kernel / sum_values
      
        return kernel

    def gaussian(self, x, y, avg_x, avg_y, sig_x, sig_y):
        value = (1 / (numpy.sqrt(2 * numpy.pi) * sig_x * sig_y)) * numpy.exp(-(((x - avg_x)**2) / (2 * sig_x) + ((y - avg_y)**2) / (2*sig_y)))
        return value

    def __blur_image(self, image, kernel):
        kernel_radius = len(kernel) // 2

        image_pad = numpy.pad(image, pad_width = kernel_radius, mode = 'constant', constant_values = 0)

        image_output = self.convolution(image_pad, kernel)

    def convolution(self, image, kernel):
        
        return 0

    def __detect_edges(self):
        pass

    def __detect_contour(self):
        pass

    def extract_features_from_images(self, space_memory):
        kernel = self.generate_kernel(3)
        for i in range(space_memory):
            image = Image.open('Batch_1/tumorImage_' + str(i + 1) + '.png')
            self.__extract_feature(numpy.asarray(image), kernel)

class TumorRecognitionBrain:
    def __init__(self):
        self.__space_memory = 1
        self.__population = Population(self.__space_memory)
        self.__learning_process = LearningProcess()
    
    def create_tumors(self):
        self.__population.creates_images()
    
    def extract_all_images(self):
        self.__learning_process.extract_features_from_images(self.__space_memory)
