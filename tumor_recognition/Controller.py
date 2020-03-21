from View import mainWindow as Mw
from Model import tumorRecognitionBrain as TrB
class tumorRecognitionApplication:
    def __init__(self):
        self.__main_window = Mw()
        self.__brain = TrB()

    def create_tumors(self):
        self.__brain.create_tumors()



if __name__ == '__main__':
    t = tumorRecognitionApplication()    
