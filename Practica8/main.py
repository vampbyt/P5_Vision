import sys
from PyQt6.QtWidgets import QApplication
from modelo import KMeansModel
from vista import KMeansView
from controlador import KMeansController

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    model = KMeansModel()
    view = KMeansView()
    controller = KMeansController(model, view)
    
    view.show()
    sys.exit(app.exec())