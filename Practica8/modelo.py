import numpy as np
import cv2
import os

class KMeansModel:
    def __init__(self):
        self.centroids = None
        self.dataset_array = None 
        self.final_labels = None  

    def entrenamiento_kmeans_centro(self, folder_path, k, total_descriptors):
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            return None, 0
            
        all_descriptors = []
        desc_per_image = max(1, total_descriptors // len(image_files))

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

        for iter_count in range(max_iters):
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
        
        self.dataset_array = dataset_array
        self.final_labels = final_labels

        labeled_data = np.hstack((dataset_array, (final_labels + 1)[:, np.newaxis]))
        
        csv_path = os.path.join(folder_path, "indexed_database.csv")
        np.savetxt(csv_path, labeled_data, fmt='%d', delimiter=',', header='R,G,B,Clase', comments='')
        
        print(f"Indexed Database guardada en: {csv_path}")

        return self.centroids, converged_iter

    def segmentar_con_imagen_prueba(self, image_path):
        if self.centroids is None:
            return None, None
            
        img = cv2.imread(image_path)
        if img is None:
            return None, None
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pixels = img.reshape((-1, 3)).astype(np.float32)
        
        distances = np.linalg.norm(pixels[:, np.newaxis] - self.centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        img_con_cajas = img.copy()
        labels_2d = labels.reshape(img.shape[:2])
        
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
                rect_x2 = min(img.shape[1], x_max - offset)
                rect_y2 = min(img.shape[0], y_max - offset)
                
                cv2.rectangle(img_con_cajas, (rect_x1, rect_y1), (rect_x2, rect_y2), color_caja, 4)
                
                # =========================================================
                # SOLUCIÓN: ETIQUETAS VISIBLES Y SEGURAS
                # =========================================================
                texto = f'C{j+1}'
                font_scale = 1.5
                grosor = 3
                
                # Calculamos cuánto mide el texto para hacerle su caja de fondo
                (w_txt, h_txt), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, font_scale, grosor)
                
                # Obligamos a que el texto empiece un poco despegado de la orilla izquierda
                text_x = max(10, rect_x1 + 10)
                
                # Calculamos la altura 'Y' desfasándola por clase para que no choquen. 
                # El 'max(h_txt + 10, ...)' garantiza que jamás será negativo (no se saldrá por arriba)
                text_y = max(h_txt + 15, rect_y1 + h_txt + 15 + (j * 45))
                
                # Garantizamos que jamás se salga por abajo
                text_y = min(text_y, img.shape[0] - 15)
                
                # Dibujamos un bloque negro sólido detrás del texto
                cv2.rectangle(img_con_cajas, 
                              (text_x - 5, text_y - h_txt - 5), 
                              (text_x + w_txt + 5, text_y + 8), 
                              (0, 0, 0), -1) 
                
                # Escribimos el texto encima del bloque negro usando el color de la clase
                cv2.putText(img_con_cajas, texto, (text_x, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_caja, grosor) 
                # =========================================================

        counts = np.bincount(labels, minlength=len(self.centroids))
        total_pixels = len(labels)
        percentages = (counts / total_pixels) * 100
        
        return img_con_cajas, percentages