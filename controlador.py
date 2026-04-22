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
            
            # 1. Pedir datos a la vista
            eta, w_inicial = self.vista.pedir_condiciones_iniciales()
            
            # 2. Configurar el modelo
            self.modelo.parametros(eta, w_inicial)
            
            # 3. Entrenar (El controlador espera a que el modelo termine)
            ha_convergido, iteracion, w_final = self.modelo.entrenar()
            
            # 4. Mostrar resultados en texto
            self.vista.mostrar_resultados(ha_convergido, iteracion, w_final)
            
            # 5. Mostrar gráfica 3D
            self.vista.dibujar_grafica_3d(
                self.modelo.X, 
                self.modelo.C_objetivo, 
                w_final, 
                eta, 
                iteracion
            )
            
            # 6. ¿Repetir?
            repetir = self.vista.preguntar_repetir()