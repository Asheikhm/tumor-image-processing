from View import MainWindow as Mw
from Model import TumorRecognitionBrain as TrB

class TumorRecognitionApplication:
    def __init__(self, view, model):
        self.__main_window = view
        self.__brain = model

    def create_tumors(self, nb_images):
        self.__brain.set_memory(nb_images)
        self.__brain.create_tumors(self.__main_window.selected_noise_Strategy())
    
    def learn(self):
        self.__brain.extract_all_images()
