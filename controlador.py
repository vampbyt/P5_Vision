from modelo import Modelo
from vista import Vista

class Controlador:
    def __init__(self):
        self.modelo = Modelo()
        self.vista = Vista()

    def ejecutar(self):
        repetir = True
        while repetir:
            self.vista.encabezado()
            
            eta, w_inicial = self.vista.pedir_condiciones_iniciales()
            
            self.modelo.parametros(eta, w_inicial)
            
            ha_convergido, iteracion, w_final = self.modelo.entrenar()
            
            self.vista.mostrar_resultados(ha_convergido, iteracion, w_final)

            self.vista.mostrar_historial(self.modelo.historial)
            
            self.vista.dibujar_grafica_3d(
                self.modelo.X, 
                self.modelo.C_objetivo, 
                w_final, 
                eta, 
                iteracion
            )
            
            repetir = self.vista.preguntar_repetir()