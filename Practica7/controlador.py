import numpy as np
from modelo import PuntoImagen, ClaseImagen, PerceptronBinario
from vista import VistaImagen
import cv2
import tkinter as tk
from tkinter import filedialog

class ControladorPerceptron:
    def __init__(self):
        self.vista = VistaImagen()
        self.modelo = None

    def preparar_datos_entrenamiento(self, muestras_crudas):
        datos = []
        etiquetas = []
        
        for clase_id, muestras in enumerate(muestras_crudas):
            pixeles = [PuntoImagen(m['rgb'], m['coords']) for m in muestras]
            clase_obj = ClaseImagen(f"Clase_{clase_id}", pixeles, self.vista.colores_bgr[clase_id])
            
            for pixel in clase_obj.pixeles:
                # Entrenamos EXCLUSIVAMENTE con Coordenadas (X, Y)
                datos.append([pixel.coordenadas[0], pixel.coordenadas[1]])
                etiquetas.append(clase_id)
                
        return np.array(datos, dtype=float), np.array(etiquetas, dtype=int)

    def dibujar_linea_divisoria(self, img_original):
        pesos, bias, _ = self.modelo.obtener_parametros()
        minimos = self.modelo.minimos
        rangos = self.modelo.rangos

        alto, ancho = img_original.shape[:2]
        
        # Trabajamos sobre una copia limpia de la imagen original
        img_resultado = img_original.copy()

        # Despejamos la ecuación matemática de la recta: Wx*X + Wy*Y + Bias = 0
        if abs(pesos[1]) > 1e-6: # Para evitar división por cero si la línea es casi vertical
            # Evaluamos la recta en los bordes izquierdo (X=0) y derecho (X=ancho) de la imagen
            x0_norm = (0 - minimos[0]) / rangos[0]
            x1_norm = (ancho - minimos[0]) / rangos[0]
            
            # Despejamos Y normalizado
            y0_norm = -(pesos[0] * x0_norm + bias) / pesos[1]
            y1_norm = -(pesos[0] * x1_norm + bias) / pesos[1]
            
            # Desnormalizamos Y para obtener el píxel exacto en la imagen
            y0 = int(y0_norm * rangos[1] + minimos[1])
            y1 = int(y1_norm * rangos[1] + minimos[1])
            
            # Dibujamos la línea recta amarilla de lado a lado
            cv2.line(img_resultado, (0, y0), (ancho, y1), (0, 255, 255), 3)
        else:
            # Caso especial: Si la línea es perfectamente vertical
            x_norm = -bias / pesos[0]
            x_val = int(x_norm * rangos[0] + minimos[0])
            cv2.line(img_resultado, (x_val, 0), (x_val, alto), (0, 255, 255), 3)
            
        return img_resultado
        
    def seleccionar_archivo(self):
        root = tk.Tk()
        root.withdraw()
        
        ruta = filedialog.askopenfilename(
            title="Selecciona la imagen",
            filetypes=[
                ("Imágenes", "*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff"),
                ("Todos los archivos", "*.*")
            ]
        )
        return ruta

    def ejecutar(self):
        while True:
            try:
                print("\nAbriendo ventana para seleccionar imagen...")
                ruta_imagen = self.seleccionar_archivo()
                
                if not ruta_imagen:
                    print("No se seleccionó ninguna imagen.")
                else:
                    print(f"Imagen seleccionada: {ruta_imagen}")
                    
                    img = self.vista.cargar_imagen(ruta_imagen)
                    
                    # Se piden los parámetros y recortamos a solo 2 pesos (X, Y)
                    eta, pesos_ini, bias_ini = self.vista.solicitar_parametros_consola()
                    pesos_ini_2d = pesos_ini[:2] 
                    
                    muestras_crudas = self.vista.obtener_muestras_usuario(img)
                    datos_x, etiquetas_y = self.preparar_datos_entrenamiento(muestras_crudas)
                    
                    print("\n--- Entrenando Modelo (Por Coordenadas) ---")
                    self.modelo = PerceptronBinario(ratio_de_aprendizaje=eta)
                    exito = self.modelo.ajustar(datos_x, etiquetas_y, pesos_iniciales=pesos_ini_2d, bias_inicial=bias_ini)
                    
                    if exito:
                        print("Trazando línea recta sobre la imagen original...")
                        img_resultado = self.dibujar_linea_divisoria(img)
                        self.vista.mostrar_resultado(img_resultado)
                    else:
                        print("\nEl proceso terminó sin trazar la línea debido a la falta de convergencia.")
                        
            except Exception as e:
                print(f"Ocurrió un error: {e}")
            
            repetir = input("\n¿Deseas intentar con otra imagen? (s/n): ").strip().lower()
            if repetir != 's':
                print("Finalizando programa. ¡Hasta luego!")
                break

if __name__ == '__main__':
    app = ControladorPerceptron()
    app.ejecutar()