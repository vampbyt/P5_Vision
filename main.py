from modelo import Modelo
from vista import Vista

if __name__ == "__main__":
    modelo = Modelo()
    vista = Vista()
    
    repetir = 's'
    while repetir.lower() == 's':
        
        # 1. Preguntar los valores de r y W
        eta_r, w_inicial = vista.pedir_condiciones_iniciales()
        
        # 2. Cargar los datos al modelo matemático
        modelo.establecer_parametros(eta_r, w_inicial)
        
        # 3. Entrenar la red
        ha_convergido, iteraciones = modelo.entrenar()
        
        # 4. Imprimir la tabla en la consola primero
        vista.imprimir_tabla_historial(modelo.historial, iteraciones, modelo.W)
        
        # 5. Abrir la gráfica 3D inmediatamente después
        print("▶ Mostrando la gráfica 3D...")
        vista.dibujar_grafica_3d(modelo.X, modelo.C_objetivo, modelo.W)
        
        # 6. Preguntar si queremos hacer otra prueba
        repetir = input("\n¿Quieres intentar con otros valores de r y W? (s/n): ")