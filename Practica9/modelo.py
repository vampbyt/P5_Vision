import numpy as np
import cv2
from collections import deque
from database_manager import BaseDatosIndexada

class ModeloVision:
    def __init__(self):
        # Variables de Entrenamiento (Region Growing)
        self.image = None
        self.seed = None
        self.threshold = 15
        self.mask = None
        
        # Variables de Clasificación
        self.clases_db = {}
        # Reglas basadas exactamente en tu BD: 
        # Cielo azul, Nubes, Montaña_piedra, Cielo, Arena, Mar, Pinos, Rocas, Bosque
        # Reglas ajustadas a: Cielo, Arena, Agua, Nubes, Rocas, Pinos, Bosque
        self.reglas_escena = {
            # --- PLAYAS ---
            frozenset(['Cielo', 'Arena', 'Agua']): 'Playa',
            frozenset(['Cielo', 'Arena']): 'Playa',
            frozenset(['Nubes', 'Arena', 'Agua']): 'Playa',
            frozenset(['Arena', 'Agua']): 'Playa',

            # --- LAGOS / LAGUNAS ---
            frozenset(['Cielo', 'Agua', 'Bosque']): 'Lago',
            frozenset(['Nubes', 'Agua', 'Bosque']): 'Lago',
            frozenset(['Cielo', 'Agua', 'Rocas']): 'Lago',
            frozenset(['Agua', 'Rocas']): 'Lago',

            # --- MONTAÑAS ---
            frozenset(['Cielo', 'Rocas', 'Bosque']): 'Montañas',
            frozenset(['Nubes', 'Rocas', 'Bosque']): 'Montañas',
            frozenset(['Cielo', 'Rocas']): 'Montañas',
            frozenset(['Rocas', 'Bosque']): 'Montañas',

            # --- BOSQUES ---
            frozenset(['Cielo', 'Bosque']): 'Bosque',
            frozenset(['Nubes', 'Bosque']): 'Bosque',
            frozenset(['Bosque']): 'Bosque'
        }

    # --- LÓGICA DE ENTRENAMIENTO (REGION GROWING) ---
    def load_image(self, path):
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"No se pudo cargar la imagen: {path}")
        self.image = img
        return self.image

    def set_seed(self, x, y):
        self.seed = (x, y)

    def set_threshold(self, threshold):
        self.threshold = threshold

    def procesar_region_growing(self):
        if self.image is None or self.seed is None:
            return None

        img = self.image.astype(np.float32)
        h, w = img.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        visited = np.zeros((h, w), bool)
        
        points_to_process = deque([self.seed])
        visited[self.seed[1], self.seed[0]] = True
        ref_color = img[self.seed[1], self.seed[0]]
        
        neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        while points_to_process:
            x, y = points_to_process.popleft()
            mask[y, x] = 255 
            
            for dx, dy in neighbors:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx]:
                    curr_color = img[ny, nx]
                    dist = np.linalg.norm(ref_color - curr_color)
                    if dist <= self.threshold:
                        points_to_process.append((nx, ny))
                        visited[ny, nx] = True 
                        
        self.mask = mask
        return self.mask
    
    def get_feature_vectors(self, n_descriptores):
        if self.mask is None: return None
        # roi contiene TODOS los píxeles (colores BGR) de la región seleccionada
        roi = self.image[self.mask == 255]
        
        if len(roi) == 0: return None

        # Si el usuario pide más descriptores de los que hay, le damos todos los posibles
        if n_descriptores >= len(roi):
            print(f"[Aviso] La región solo tiene {len(roi)} píxeles válidos. Extrayendo todos.")
            return np.round(roi, 2)
        else:
            # Seleccionar N píxeles de forma aleatoria (sin repetir) de la región
            indices = np.random.choice(len(roi), size=n_descriptores, replace=False)
            descriptores_seleccionados = roi[indices]
            return np.round(descriptores_seleccionados, 2)

    def guardar_en_bd(self, nombre_clase, vectores):
        db = BaseDatosIndexada()
        # Ahora pasamos la lista completa de vectores, no solo uno
        db.guardar_descriptores(nombre_clase, vectores)
        db.cerrar()

    # --- LÓGICA DE CLASIFICACIÓN ---
    def cargar_bd_para_clasificar(self):
        db = BaseDatosIndexada()
        self.clases_db = db.cargar_todos_los_descriptores()
        db.cerrar()

    def clasificar_ventana(self, roi_img):
        """Calcula el vector de la ventana y aplica K-NN (K=1) Euclidiano"""
        vector_ventana = np.mean(roi_img, axis=(0, 1))
        min_dist = float('inf')
        mejor_clase = "Desconocida"
        
        for nombre_clase, vectores in self.clases_db.items():
            for v in vectores:
                dist = np.linalg.norm(vector_ventana - v)
                if dist < min_dist:
                    min_dist = dist
                    mejor_clase = nombre_clase
        return mejor_clase

    def determinar_escena_final(self, subclases_detectadas):
        mejor_escena = "Escena no identificada"
        for sub_conjunto, nombre_escena in self.reglas_escena.items():
            if sub_conjunto.issubset(subclases_detectadas):
                mejor_escena = nombre_escena
                break
        return mejor_escena
    
    def obtener_nombres_clases(self):
        db = BaseDatosIndexada()
        # Carga el diccionario y extrae solo las llaves (los nombres)
        clases = list(db.cargar_todos_los_descriptores().keys())
        db.cerrar()
        return clases