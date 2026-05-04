import cv2
import numpy as np
import random

# ==========================================
# 1. MODELO DEL PERCEPTRÓN
# ==========================================
class ModeloPerceptron:
    def __init__(self, X, C_objetivo, eta, w_inicial):
        self.X = np.array(X)
        self.C_objetivo = np.array(C_objetivo)
        self.eta = eta
        self.W = np.array(w_inicial, dtype=float)

    def entrenar(self):
        iteracion = 0
        ha_convergido = False
        num_patrones = len(self.X)

        print("\n--- Iniciando Entrenamiento del Perceptrón ---")
        while not ha_convergido and iteracion < 1000:
            iteracion += 1
            errores = 0
            
            for i in range(num_patrones):
                x = self.X[i]
                c = self.C_objetivo[i]
                
                # Cálculo de Net (Producto punto)
                red = np.dot(self.W, x)
                
                # REGLA ESTRICTA 
                if c == 0 and red >= 0:
                    errores += 1
                    self.W = self.W - (self.eta * x)
                elif c == 1 and red <= 0:
                    errores += 1
                    self.W = self.W + (self.eta * x)

            if errores == 0:
                ha_convergido = True

        return ha_convergido, iteracion, self.W

# ==========================================
# 2. PROCESAMIENTO Y SEGMENTACIÓN DE IMAGEN
# ==========================================
def main():
    # 1. Cargar y redimensionar la imagen
    ruta_imagen = 'R.jpg' # CAMBIA ESTO POR EL NOMBRE DE TU IMAGEN
    img = cv2.imread(ruta_imagen)
    
    if img is None:
        print(f"Error: No se pudo cargar la imagen {ruta_imagen}")
        return

    alto_maximo = 600
    alto_original, ancho_original = img.shape[:2]
    if alto_original > alto_maximo:
        proporcion = alto_maximo / float(alto_original)
        nuevo_ancho = int(ancho_original * proporcion)
        img = cv2.resize(img, (nuevo_ancho, alto_maximo))

    img_display = img.copy()

    # 2. Configuración Inicial
    print("=== CONFIGURACIÓN DEL PERCEPTRÓN ===")
    eta = float(input("Ingresa la tasa de aprendizaje (ej. 0.1): "))
    
    # Bucle para asegurar que siempre se ingresen exactamente 4 pesos
    while True:
        w_input = input("Ingresa los 4 pesos iniciales separados por espacio (ej. 0 0 0 0): ")
        w_inicial = [float(x) for x in w_input.split()]
        
        if len(w_inicial) == 4:
            break # Si son 4, salimos del bucle y continuamos
        else:
            print(f"Error: Ingresaste {len(w_inicial)} valores. Deben ser exactamente 4. Inténtalo de nuevo.")
    
    num_clases = int(input("¿Cuántas clases vas a seleccionar? (Ingresa 2): "))
    if num_clases != 2:
        num_clases = 2

    X_entrenamiento = []
    Y_entrenamiento = []
    
    # Colores en formato BGR de OpenCV: (Azul, Verde, Rojo)
    # Clase 0: Rojo, Clase 1: Azul
    colores_bgr = [(0, 0, 255), (255, 0, 0)] 
    
    # 3. Selección de Ventanas y Extracción de Características
    for clase_id in range(num_clases):
        print(f"\n--- Clase {clase_id} ---")
        puntos_por_ventana = int(input(f"¿Cuántos representantes (descriptores) quieres para la Clase {clase_id}?: "))
        
        print("Dibuja un rectángulo con el mouse y presiona 'ENTER' para confirmar.")
        roi = cv2.selectROI("Seleccion de Ventanas", img_display, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Seleccion de Ventanas")
        
        x_min, y_min, ancho, alto = roi
        
        # Dibujar el rectángulo
        cv2.rectangle(img_display, (x_min, y_min), (x_min+ancho, y_min+alto), colores_bgr[clase_id], 2)
        
        # Generar puntos aleatorios y extraer color
        for _ in range(puntos_por_ventana):
            rand_x = random.randint(x_min, x_min + ancho - 1)
            rand_y = random.randint(y_min, y_min + alto - 1)
            
            # Dibujar el punto en la imagen
            cv2.circle(img_display, (rand_x, rand_y), 3, colores_bgr[clase_id], -1)
            
            b, g, r = img[rand_y, rand_x]
            
            # Normalización
            r_norm = r / 255.0
            g_norm = g / 255.0
            b_norm = b / 255.0
            bias = 1.0
            
            X_entrenamiento.append([r_norm, g_norm, b_norm, bias])
            Y_entrenamiento.append(clase_id)

    # Mostrar la imagen original con las ventanas y puntos
    cv2.imshow("Original con Muestras", img_display)
    cv2.waitKey(500) # Pausa breve antes de entrenar

    # 4. Entrenamiento
    perceptron = ModeloPerceptron(X_entrenamiento, Y_entrenamiento, eta, w_inicial)
    exito, iteraciones, pesos_finales = perceptron.entrenar()
    
    if exito:
        print(f"\n¡ÉXITO! El perceptrón convergió en {iteraciones} iteraciones.")
        print(f"Pesos finales (W): {pesos_finales}")
        
        # ==========================================
        # 5. SEGMENTACIÓN DE TODA LA IMAGEN (TIPO MÁSCARA)
        # ==========================================
        print("Clasificando la imagen completa. Por favor espera un momento...")
        
        # Separar canales de la imagen completa y normalizarlos
        b = img[:, :, 0] / 255.0
        g = img[:, :, 1] / 255.0
        r = img[:, :, 2] / 255.0
        
        # Calcular el producto punto (Net) para toda la matriz de la imagen a la vez
        # W[0]*R + W[1]*G + W[2]*B + W[3]*Bias
        net_imagen = (pesos_finales[0] * r) + (pesos_finales[1] * g) + (pesos_finales[2] * b) + pesos_finales[3]
        
        # Crear una imagen negra del mismo tamaño que la original
        img_segmentada = np.zeros_like(img)
        
        # Según la regla del perceptrón:
        # Clase 0 (Rojo) fue forzada a tener valores Net negativos (<= 0)
        # Clase 1 (Azul) fue forzada a tener valores Net positivos (> 0)
        img_segmentada[net_imagen <= 0] = colores_bgr[0] # Pintar de Rojo
        img_segmentada[net_imagen > 0] = colores_bgr[1]  # Pintar de Azul
        
        # Mostrar los resultados
        cv2.imshow("Imagen Segmentada por el Perceptron", img_segmentada)
        print("Presiona cualquier tecla en las ventanas de las imágenes para salir.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print(f"\nEl perceptrón no convergió tras {iteraciones} iteraciones. Clases no separables.")
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()