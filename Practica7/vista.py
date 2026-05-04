import cv2
import random

class VistaImagen:
    def __init__(self, alto_maximo=600):
        self.alto_maximo = alto_maximo
        self.colores_bgr = [(0, 0, 255), (255, 0, 0)] # Rojo, Azul

    def cargar_imagen(self, ruta):
        img = cv2.imread(ruta)
        if img is None:
            raise FileNotFoundError(f"No se pudo cargar la imagen: {ruta}")
        
        # Redimensionar si es muy grande
        alto_original, ancho_original = img.shape[:2]
        if alto_original > self.alto_maximo:
            proporcion = self.alto_maximo / float(alto_original)
            nuevo_ancho = int(ancho_original * proporcion)
            img = cv2.resize(img, (nuevo_ancho, self.alto_maximo))
        return img

    def solicitar_parametros_consola(self):
        print("\n=== CONFIGURACIÓN DEL MODELO ===")
        eta = float(input("Ingresa la tasa de aprendizaje (ej. 0.1): "))
        while True:
            w_input = input("Ingresa 3 pesos iniciales y el bias separados por espacio (ej. 0 0 0 0): ")
            w_inicial = [float(x) for x in w_input.split()]
            if len(w_inicial) == 4:
                return eta, w_inicial[:3], w_inicial[3]
            print("Error: Deben ser exactamente 4 valores. Inténtalo de nuevo.")

    def obtener_muestras_usuario(self, img, num_clases=2):
        img_display = img.copy()
        muestras_por_clase = []

        for clase_id in range(num_clases):
            print(f"\n--- Selección para la Clase {clase_id} ---")
            puntos = int(input(f"¿Cuántos representantes quieres para la Clase {clase_id}?: "))
            
            print("Dibuja un rectángulo y presiona ENTER.")
            roi = cv2.selectROI("Seleccion de ROI", img_display, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow("Seleccion de ROI")
            
            x_min, y_min, ancho, alto = roi
            cv2.rectangle(img_display, (x_min, y_min), (x_min+ancho, y_min+alto), self.colores_bgr[clase_id], 2)
            
            puntos_extraidos = []
            for _ in range(puntos):
                rx = random.randint(x_min, x_min + ancho - 1)
                ry = random.randint(y_min, y_min + alto - 1)
                cv2.circle(img_display, (rx, ry), 3, self.colores_bgr[clase_id], -1)
                
                # Guardar color [R, G, B] y coordenadas
                b, g, r = img[ry, rx]
                puntos_extraidos.append({'rgb': [r, g, b], 'coords': (rx, ry)})
                
            muestras_por_clase.append(puntos_extraidos)

        cv2.imshow("Muestras Seleccionadas", img_display)
        cv2.waitKey(1000)
        cv2.destroyWindow("Muestras Seleccionadas")
        return muestras_por_clase

    def mostrar_resultado(self, imagen_segmentada):
        cv2.imshow("Imagen Segmentada (MVC)", imagen_segmentada)
        print("\nPresiona cualquier tecla en la ventana de la imagen para finalizar.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()