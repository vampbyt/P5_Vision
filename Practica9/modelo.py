import numpy as np
import cv2
from collections import deque
from database_manager import BaseDatosIndexada

class ModeloVision:
    def __init__(self):
        self.image = None
        self.seeds = [] # AHORA ES UNA LISTA PARA MÚLTIPLES PUNTOS
        self.threshold = 15
        self.mask = None # La máscara final fusionada
        
        self.clases_db = {}
        
        self.reglas_escena = {
            frozenset(['Cielo', 'Arena', 'Agua']): 'Playa',
            frozenset(['Nubes', 'Arena', 'Agua']): 'Playa',
            frozenset(['Arena', 'Agua']): 'Playa',

            frozenset(['Cielo', 'Agua', 'Bosque']): 'Lago',
            frozenset(['Nubes', 'Agua', 'Bosque']): 'Lago',
            frozenset(['Cielo', 'Agua', 'Rocas']): 'Lago',

            frozenset(['Cielo', 'Rocas', 'Bosque']): 'Montañas',
            frozenset(['Nubes', 'Rocas', 'Bosque']): 'Montañas',
            frozenset(['Cielo', 'Rocas']): 'Montañas',
            frozenset(['Rocas', 'Bosque']): 'Montañas',

            frozenset(['Cielo', 'Bosque']): 'Bosque',
            frozenset(['Nubes', 'Bosque']): 'Bosque'
        }

    def load_image(self, path):
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"No se pudo cargar la imagen: {path}")
        self.image = img
        self.seeds = [] # Limpiar semillas al cargar nueva imagen
        self.mask = None
        return self.image

    def agregar_semilla(self, x, y):
        self.seeds.append((x, y))

    def limpiar_semillas(self):
        self.seeds = []
        self.mask = None

    def set_threshold(self, threshold):
        self.threshold = threshold

    def procesar_region_growing(self):
        """
        Crece múltiples semillas, calcula sus colores y usa CAJ para
        fusionar las máscaras que sean matemáticamente similares.
        """
        if self.image is None or not self.seeds:
            return None

        img = self.image.astype(np.float32)
        h, w = img.shape[:2]
        
        mascaras_individuales = []
        vectores_individuales = []
        neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        # 1. Crecer cada semilla en su propia máscara aislada
        for sx, sy in self.seeds:
            mask_temp = np.zeros((h, w), np.uint8)
            visited = np.zeros((h, w), bool)
            points_to_process = deque([(sx, sy)])
            visited[sy, sx] = True
            ref_color = img[sy, sx]
            
            while points_to_process:
                x, y = points_to_process.popleft()
                mask_temp[y, x] = 255 
                
                for dx, dy in neighbors:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx]:
                        curr_color = img[ny, nx]
                        dist = np.linalg.norm(ref_color - curr_color)
                        if dist <= self.threshold:
                            points_to_process.append((nx, ny))
                            visited[ny, nx] = True 
                            
            mascaras_individuales.append(mask_temp)
            
            # Calcular el color de esta máscara para la CAJ
            roi = self.image[mask_temp == 255]
            if len(roi) > 0:
                vectores_individuales.append(np.mean(roi, axis=0))
            else:
                vectores_individuales.append(ref_color)

        # 2. CLASIFICACIÓN ASCENDENTE JERÁRQUICA (Fusión de Máscaras)
        # Comparamos todas las regiones contra la primera semilla que pusiste
        self.mask = np.zeros((h, w), np.uint8)
        umbral_caj = 40.0 # Tolerancia de fusión entre regiones
        
        if len(vectores_individuales) > 0:
            vector_base = vectores_individuales[0]
            for i, vector_actual in enumerate(vectores_individuales):
                distancia = np.linalg.norm(vector_base - vector_actual)
                # Si la región es parecida a la principal, la fusionamos
                if distancia <= umbral_caj:
                    self.mask = cv2.bitwise_or(self.mask, mascaras_individuales[i])

        return self.mask

    # ... (El resto de tus funciones como get_feature_vectors, guardar_en_bd, 
    # clasificar_vector y determinar_escena_final se quedan EXACTAMENTE igual) ...
    def get_feature_vectors(self, n_descriptores):
        if self.mask is None: return None
        roi = self.image[self.mask == 255]
        
        if len(roi) == 0: return None

        if n_descriptores >= len(roi):
            print(f"[Aviso] La región solo tiene {len(roi)} píxeles válidos. Extrayendo todos.")
            return np.round(roi, 2)
        else:
            indices = np.random.choice(len(roi), size=n_descriptores, replace=False)
            descriptores_seleccionados = roi[indices]
            return np.round(descriptores_seleccionados, 2)

    def guardar_en_bd(self, nombre_clase, vectores):
        db = BaseDatosIndexada()
        db.guardar_descriptores(nombre_clase, vectores)
        db.cerrar()

    def obtener_nombres_clases(self):
        db = BaseDatosIndexada()
        clases = list(db.cargar_todos_los_descriptores().keys())
        db.cerrar()
        return clases

    def cargar_bd_para_clasificar(self):
        db = BaseDatosIndexada()
        self.clases_db = db.cargar_todos_los_descriptores()
        db.cerrar()

    def clasificar_vector(self, vector):
        min_dist = float('inf')
        mejor_clase = "Desconocida"
        
        for nombre_clase, vectores in self.clases_db.items():
            for v in vectores:
                dist = np.linalg.norm(vector - v)
                if dist < min_dist:
                    min_dist = dist
                    mejor_clase = nombre_clase
        return mejor_clase


    # --- CLASIFICACIÓN ASCENDENTE JERÁRQUICA (Fase 2) ---
    def clasificacion_ascendente_jerarquica(self, lista_vectores, umbral_fusion=40.0):
        """
        Recibe los vectores de las ventanas del usuario y fusiona los similares.
        """
        if not lista_vectores:
            return []

        clusteres = [{'id': i, 'vector': vec, 'peso': 1} for i, vec in enumerate(lista_vectores)]
        
        hubo_fusion = True
        while hubo_fusion and len(clusteres) > 1:
            hubo_fusion = False
            min_dist = float('inf')
            mejor_par = None
            
            for i in range(len(clusteres)):
                for j in range(i + 1, len(clusteres)):
                    dist = np.linalg.norm(clusteres[i]['vector'] - clusteres[j]['vector'])
                    if dist < min_dist:
                        min_dist = dist
                        mejor_par = (i, j)
            
            if mejor_par is not None and min_dist <= umbral_fusion:
                idx1, idx2 = mejor_par
                c1 = clusteres[idx1]
                c2 = clusteres[idx2]
                
                nuevo_peso = c1['peso'] + c2['peso']
                nuevo_vector = ((c1['vector'] * c1['peso']) + (c2['vector'] * c2['peso'])) / nuevo_peso
                
                nuevo_cluster = {'id': c1['id'], 'vector': nuevo_vector, 'peso': nuevo_peso}
                
                clusteres.pop(max(idx1, idx2))
                clusteres.pop(min(idx1, idx2))
                clusteres.append(nuevo_cluster)
                
                hubo_fusion = True 

        return [c['vector'] for c in clusteres]

    def determinar_escena_final(self, subclases_detectadas):
        mejor_escena = "Escena no identificada"
        for sub_conjunto, nombre_escena in self.reglas_escena.items():
            if sub_conjunto.issubset(subclases_detectadas):
                mejor_escena = nombre_escena
                break
        return mejor_escena