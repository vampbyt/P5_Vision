import numpy as np
import cv2
import os

class KMeansModel:
    def __init__(self):
        self.centroids = None
        self.dataset_array = None 
        self.final_labels = None  
        self.num_representantes = 0 

    def entrenamiento_kmeans_centro(self, folder_path, k, descriptores_por_clase):
        self.num_representantes = descriptores_por_clase 
        
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            return None, 0, None
            
        all_descriptors = []
        
        pool_size = max(50000, k * descriptores_por_clase * 5)
        desc_per_image = max(1, pool_size // len(image_files))

        for file_name in image_files:
            img_path = os.path.join(folder_path, file_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pixels = img.reshape((-1, 3)).astype(np.float32)
            
            if desc_per_image > len(pixels):
                indices = np.arange(len(pixels))
            else:
                indices = np.random.choice(pixels.shape[0], desc_per_image, replace=False)
                
            all_descriptors.append(pixels[indices])

        dataset_array = np.vstack(all_descriptors)

        centroid_indices = np.random.choice(dataset_array.shape[0], k, replace=False)
        centroids = dataset_array[centroid_indices]

        max_iters = 100
        converged_iter = max_iters
        history_centroids = []

        for iter_count in range(max_iters):
            history_centroids.append(centroids.copy())
            
            distances = np.linalg.norm(dataset_array[:, np.newaxis] - centroids, axis=2)
            labels = np.argmin(distances, axis=1)

            new_centroids = np.zeros_like(centroids)
            for j in range(k):
                points_in_cluster = dataset_array[labels == j]
                if len(points_in_cluster) > 0:
                    new_centroids[j] = points_in_cluster.mean(axis=0)
                else:
                    new_centroids[j] = centroids[j]

            if np.all(centroids == new_centroids):
                converged_iter = iter_count + 1
                break
                
            centroids = new_centroids

        self.centroids = centroids

        final_distances = np.linalg.norm(dataset_array[:, np.newaxis] - self.centroids, axis=2)
        final_labels = np.argmin(final_distances, axis=1)
        
        balanced_data = []
        balanced_labels = []
        
        for j in range(k):
            puntos_clase = dataset_array[final_labels == j]
            
            if len(puntos_clase) > 0:
                replace_flag = len(puntos_clase) < descriptores_por_clase
                indices = np.random.choice(len(puntos_clase), descriptores_por_clase, replace=replace_flag)
                balanced_data.append(puntos_clase[indices])
            else:
                balanced_data.append(np.tile(self.centroids[j], (descriptores_por_clase, 1)))
                
            balanced_labels.append(np.full(descriptores_por_clase, j))
            
        self.dataset_array = np.vstack(balanced_data)
        self.final_labels = np.concatenate(balanced_labels)

        labeled_data = np.hstack((self.dataset_array, (self.final_labels + 1)[:, np.newaxis]))
        
        csv_path = os.path.join(folder_path, "indexed_database.csv")
        np.savetxt(csv_path, labeled_data, fmt='%d', delimiter=',', header='R,G,B,Clase', comments='')
        
        print(f"Indexed Database guardada en: {csv_path}")

        return self.centroids, converged_iter, history_centroids

    def segmentar_imagen_con_puntos(self, img_bgr, puntos_y, puntos_x):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pixels = img_rgb.reshape((-1, 3)).astype(np.float32)
        
        # Medir distancias para toda la imagen para poder dibujar las cajas
        distances = np.linalg.norm(pixels[:, np.newaxis] - self.centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        img_con_cajas = img_rgb.copy()
        labels_2d = labels.reshape(img_rgb.shape[:2])
        
        # 1. Dibujar Cajas y Textos (Misma lógica segura)
        for j in range(len(self.centroids)):
            y_indices, x_indices = np.where(labels_2d == j)
            
            if len(x_indices) > 100: 
                x_min = int(np.percentile(x_indices, 2))
                x_max = int(np.percentile(x_indices, 98))
                y_min = int(np.percentile(y_indices, 2))
                y_max = int(np.percentile(y_indices, 98))
                
                color_caja = (int(self.centroids[j][0]), int(self.centroids[j][1]), int(self.centroids[j][2]))
                offset = j * 8
                
                rect_x1 = max(0, x_min + offset)
                rect_y1 = max(0, y_min + offset)
                rect_x2 = min(img_rgb.shape[1], x_max - offset)
                rect_y2 = min(img_rgb.shape[0], y_max - offset)
                
                cv2.rectangle(img_con_cajas, (rect_x1, rect_y1), (rect_x2, rect_y2), color_caja, 4)
                
                texto = f'C{j+1}'
                font_scale = 1.5
                grosor = 3
                
                (w_txt, h_txt), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, font_scale, grosor)
                text_x = max(10, rect_x1 + 10)
                text_y = max(h_txt + 15, rect_y1 + h_txt + 15 + (j * 45))
                text_y = min(text_y, img_rgb.shape[0] - 15)
                
                cv2.rectangle(img_con_cajas, (text_x - 5, text_y - h_txt - 5), (text_x + w_txt + 5, text_y + 8), (0, 0, 0), -1) 
                cv2.putText(img_con_cajas, texto, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_caja, grosor) 

        # 2. Pintar únicamente los puntos al azar que nos mandó el controlador y contarlos
        puntos_por_clase = np.zeros(len(self.centroids), dtype=int)
        
        for i in range(len(puntos_x)):
            px = puntos_x[i]
            py = puntos_y[i]
            
            # Revisamos qué clase le tocó a esa coordenada específica
            clase_asignada = labels_2d[py, px]
            
            # Sumamos al contador
            puntos_por_clase[clase_asignada] += 1
            
            # Sacamos el color de esa clase
            color_punto = (int(self.centroids[clase_asignada][0]), 
                           int(self.centroids[clase_asignada][1]), 
                           int(self.centroids[clase_asignada][2]))
            
            # Repintamos el puntito blanco original con su color matemático
            cv2.circle(img_con_cajas, (px, py), radius=3, color=(0, 0, 0), thickness=-1)
            cv2.circle(img_con_cajas, (px, py), radius=2, color=color_punto, thickness=-1)

        return img_con_cajas, puntos_por_clase