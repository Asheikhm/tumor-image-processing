import Controller
import View
import Model
import numpy

def build_application():
    model = Model.TumorRecognitionBrain()
    view = View.MainWindow()
    controller = Controller.TumorRecognitionApplication(view, model)

    prepare_button = View.PrepareButtonView('Prepare', controller)
    select_button = View.SelectButtonView('Select', controller)
    learn_button = View.LearnButtonView('Learn', controller)
    
    view.menu().initialize_buttons(select_button, prepare_button, learn_button)
    view.initUI()

if __name__ == '__main__':
    #build_application()

    a = numpy.array([[1, 2], [3, 4], [5,6]])
    for nindex, x in numpy.ndenumerate(a):
        print(x)
