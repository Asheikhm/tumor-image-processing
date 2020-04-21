import math
import numpy
import random
import time
import os
import Strategy
from numpy.fft import fft, ifft, fft2, ifft2, fftshift

from PIL import Image, ImageFilter


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
    def __init__(self, tumor_parameters, width_image = 500, height_image = 500):
        self.__tumor_parameters = tumor_parameters
        self.__width = width_image
        self.__height = height_image
        self.__pixels = numpy.zeros((width_image,height_image))

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
        self.__tumor_parameters.amplitude = random.randrange(100,255)
        self.__set_gaussian_coefficients()

    def gaussian_function(self, x, y):
        f = self.__tumor_parameters.amplitude * (numpy.exp(-(self.__coeff_a*((x - self.__tumor_parameters.avg_x)**2) + 2*self.__coeff_b*(x - self.__tumor_parameters.avg_x)*(y - self.__tumor_parameters.avg_y) + self.__coeff_c*((y - self.__tumor_parameters.avg_y)**2))))
        return f

    def __set_gaussian_coefficients(self):
        self.__coeff_a = (math.cos(self.__tumor_parameters.theta)**2) / (2*(self.__tumor_parameters.sig_x**2)) + (math.sin(self.__tumor_parameters.theta)**2) / (2*(self.__tumor_parameters.sig_y**2))
       
        self.__coeff_b = (math.sin(2*self.__tumor_parameters.theta)) / (4*(self.__tumor_parameters.sig_x**2)) + (math.sin(2*self.__tumor_parameters.theta)) / (4*(self.__tumor_parameters.sig_y**2))
       
        self.__coeff_c = (math.sin(self.__tumor_parameters.theta)**2) / (2*(self.__tumor_parameters.sig_x**2)) + (math.cos(self.__tumor_parameters.theta)**2) / (2*(self.__tumor_parameters.sig_y**2))

    def convert_to_png_file(self, folder_name):
        im = Image.fromarray(self.__image.get_pixels())
        name = 'tumorImage_'+ str(self.__image_name)+'.png'
        currentDirectory = os.getcwd() + '/'+ folder_name + '/' + name
        im.save(currentDirectory,'PNG')
    
    def apply_noise(self, appropriate_strategy):
        appropriate_strategy.apply_noise(self.__image)

class Population:
    def __init__(self, name):
        self.__folder_name = name
        self.__image_generators = []
        self.__noise_strategy = {
            1 : Strategy.SaltNPepperStrategy(),
            2 : Strategy.GaussianStrategy()
        }
    
    def creates_images(self, selected_strategy):
        #Creates folder if it doesn't exist
        if not os.path.exists(self.__folder_name):
            os.makedirs(self.__folder_name)

        choosen_strategy = self.__noise_strategy[selected_strategy]
        for image_generator in self.__image_generators:
            image_generator.set_parameters()
            image_generator.create_image()
            image_generator.apply_noise(choosen_strategy)
            image_generator.convert_to_png_file(self.__folder_name)

    def populate(self, nb_images):
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
        self.__filter_strategy = {
            1 : Strategy.MedianFilterStrategy(),
            2 : Strategy.GaussianFilterStrategy()
        }

    def __extract_feature(self, image):
        pass

    def convert_array_to_png_file(self, im_array, index):
        rescaled = (255.0 / im_array.max() * (im_array - im_array.min())).astype(numpy.uint8)

        im = Image.fromarray(rescaled)
        im.save('t_' + str(index) + '.png')


    def __detect_edges(self, image, y_direction_kernel, x_direction_kernel):
        image_size = image.shape[0]
        kernel_size = y_direction_kernel.shape[0]
        binary_threshold = 80 

        image[image > binary_threshold] = 255
        image[image <= binary_threshold] = 0

        Image.fromarray(image).show('threshold')
        image_output = numpy.zeros((image_size, image_size))
        image_padded = numpy.pad(image, pad_width = kernel_size // 2, mode = 'constant', constant_values = 0)

        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                submatrix = image_padded[x : x + kernel_size, y : y + kernel_size]

                vertical_transformed_pixels = x_direction_kernel * submatrix
                vertical_score = (vertical_transformed_pixels.sum() + 4) / 8

                horizontal_transformed_pixels = y_direction_kernel * submatrix
                horizontal_score = horizontal_transformed_pixels.sum() / 4

                edge_score = (vertical_score**2 + horizontal_score **2) ** 0.5
                image_output[x][y] = edge_score * 3
                
        im = Image.fromarray(image_output)
        im.show()
        return image_output

    def __detect_contour(self, image):
        image = Image.fromarray(image).convert('L')
        image2 = image.filter(ImageFilter.CONTOUR)
        image2.show('contour')


    def extract_features_from_images(self, space_memory, selected_strategy):
        #Kernels for edge detection
        y_direction_kernel = numpy.array([(-1, -2, -1),(0, 0, 0),(1, 2, 1)])
        x_direction_kernel = numpy.array([(-1, 0, 1),(-2, 0, 2),(-1, 0, 1)])
        
        choosen_strategy = self.__filter_strategy[selected_strategy]

        for i in range(space_memory):
            image = Image.open('Batch_1/tumorImage_' + str(i + 1) + '.png')
            #image = Image.open('gaussian_.png').convert('L')
            image = choosen_strategy.apply_filter(image)
            image_edges = self.__detect_edges(image, y_direction_kernel, x_direction_kernel)
            self.__detect_contour(image_edges)

class TumorRecognitionBrain:
    def __init__(self, nb_images = 1):
        self.__space_memory = nb_images
        self.__population1 = Population('Batch_1')
        self.__population2 = Population('Batch_2')
        self.__learning_process = LearningProcess()
        self.__selected_strategy = 2 #Gaussian default
    
    def create_tumors(self, selected_strategy):
        self.__selected_strategy = selected_strategy
        self.__population1.creates_images(selected_strategy)
        self.__population2.creates_images(selected_strategy)

    
    def set_memory(self, nb_images):
        self.__space_memory = nb_images

        #First population is for the computer to learn based on theses images
        self.__population1.populate(self.__space_memory)

        #Second population is to test the computer's knowledge
        self.__population2.populate(self.__space_memory // 2) 

    
    def extract_all_images(self):
        self.__learning_process.extract_features_from_images(self.__space_memory, self.__selected_strategy)