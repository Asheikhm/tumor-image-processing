from View import MainWindow as Mw
from Model import TumorRecognitionBrain as TrB

class TumorRecognitionApplication:
    def __init__(self, view, model):
        self.__main_window = view
        self.__brain = model

    def create_tumors(self):
        self.__brain.create_tumors()
    
    def learn(self):
        self.__brain.extract_all_images()
