from abc import ABC, abstractmethod 
import numpy
from numpy.fft import fft, ifft, fft2, ifft2, fftshift

from PIL import ImageFilter, Image


class NoiseStrategy(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def apply_noise(self, image):
        pass

class GaussianStrategy(NoiseStrategy):
    def __init__(self):
        pass

    def apply_noise(self, image):
        image_array = image.get_pixels()
        mean = 0
        var = 100
        sigma = var ** 0.5
        gaussian = numpy.random.normal(mean, sigma, (image_array.shape[0], image_array.shape[1]))
        noisy = image_array + gaussian
        noisy[noisy > 255] = 255
        noisy[noisy < 0] = 0
        noisy = noisy.astype(numpy.uint8)
        image.set_pixels(noisy)

class SaltNPepperStrategy(NoiseStrategy):
    def __init__(self):
        pass
    
    def apply_noise(self, image):
        image_array = len(image.get_pixels())
        rnd = numpy.random.rand(image_array, image_array)
        image.get_pixels()[rnd < 0.004] = 0
        image.get_pixels()[rnd > 1 - 0.004] = 255

        image.set_pixels(image.get_pixels().astype(numpy.uint8))

class FilterNoiseStrategy(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def apply_filter(self, image, kernel):
        pass
 
class GaussianFilterStrategy(FilterNoiseStrategy):
    def __init__(self):
        pass

    def apply_filter(self, image):
        image = numpy.array(image)

        kernel = self.__generate_kernel()
        kernel_size = kernel.shape[0]
        kernel_radius = kernel_size // 2
        image_size = image.shape[0]
        image_output = numpy.zeros((image_size, image_size))

        #Add zeros around the image for the submatrices
        image_padded = numpy.pad(image, pad_width = kernel_radius, mode = 'constant', constant_values = 0)
        submatrix = numpy.zeros((kernel_size, kernel_size))

        image_output = numpy.real(ifft2(fft2(image_padded)*fft2(kernel, s=image_padded.shape)))

        #Create array that contains submatrices
        # for x in range(image.shape[0]):
        #     for y in range(image.shape[1]):
        #         submatrix = image_padded[x : x + kernel_size, y : y + kernel_size]
        #         self.convolution(image_output, x, y, kernel, submatrix)
        # for indexes, value in numpy.ndenumerate(image):
        #     submatrix = image_padded[indexes[0] : indexes[0] + kernel_size, indexes[1] : indexes[1] + kernel_size]
        #     self.convolution(image_output, indexes, kernel, submatrix)

        # image_output = image_output/ 255.0
        im = Image.fromarray(image_output)
        im.show()
        return image_output

    def convolution(self, image_output, indexes, kernel, submatrix):
        image_output[indexes[0]][indexes[1]] = numpy.multiply(submatrix, kernel).sum()

    def __gaussian(self, x, y, avg_x, avg_y, sig_x, sig_y):
        value = (1 / (numpy.sqrt(2 * numpy.pi) * sig_x * sig_y)) * numpy.exp(-(((x - avg_x) ** 2) / (2 * sig_x) + ((y - avg_y) ** 2) / (2 * sig_y)))
        return value

    def __generate_kernel(self):
        kernel = numpy.zeros((3,3))
        sig_x = (3 - 1) / 5
        sig_y = sig_x

        avg_x = ((3 - 1) / 2)
        avg_y = avg_x

        #populate kernel
        rows, cols = kernel.shape
        for i in range(rows):
            x = numpy.ones((cols)) * i 
            y = numpy.arange(cols)
            kernel[i,:] = self.__gaussian(x, y, avg_x, avg_y, sig_x, sig_y)

        #normalize kernel
        sum_values = numpy.sum(kernel)
        kernel = kernel / sum_values

        return kernel
    
class MedianFilterStrategy(FilterNoiseStrategy):
    def __init__(self):
        pass
    
    def apply_filter(self, image):
        image = image.filter(ImageFilter.MedianFilter(size = 3)) 
        image = numpy.array(image)

        return image
    
