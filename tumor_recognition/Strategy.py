from abc import ABC, abstractmethod 
import numpy

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
        mean = 80
        var = 10
        sigma = var ** 0.5
        gaussian = numpy.random.normal(mean, sigma, (image_array.shape[0], image_array.shape[1]))
        noise = image_array + gaussian
        image.set_pixels(noise)

class SaltNPepperStrategy(NoiseStrategy):
    def __init__(self):
        pass
    
    def apply_noise(self, image):
        image_array = len(image.get_pixels())
        rnd = numpy.random.rand(image_array, image_array)
        image.get_pixels()[rnd < 0.004] = 0
        image.get_pixels()[rnd > 1 - 0.004] = 255

class FilterNoiseStrategy(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def apply_filter(self, image):
        pass

class GaussianFilterStrategy(FilterNoiseStrategy):
    def __init__(self):
        pass

    def apply_filter(self, image):
        pass

class SaltNPepperFilterStrategy(FilterNoiseStrategy):
    def __init__(self):
        pass
    
    def apply_filter(self, image):
        pass
    