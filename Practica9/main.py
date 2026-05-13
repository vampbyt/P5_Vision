from modelo import ModeloVision
from vista import VistaVision
from controlador import ControladorVision

if __name__ == "__main__":
    modelo = ModeloVision()
    vista = VistaVision()
    controlador = ControladorVision(modelo, vista)
    
    controlador.iniciar()