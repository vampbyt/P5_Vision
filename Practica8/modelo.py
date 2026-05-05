import numpy as np
import cv2
import os

class KMeansModel:
    def __init__(self):
        self.centroids = None

    def train_kmeans_on_dataset(self, folder_path, k, total_descriptors):
        # 1. Leer imágenes (No indexed database)
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

        # 2. Proceso de entrenamiento K-Means manual
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

        # 3. Creación de la Indexed Database
        final_distances = np.linalg.norm(dataset_array[:, np.newaxis] - self.centroids, axis=2)
        final_labels = np.argmin(final_distances, axis=1)

        indexed_folder = os.path.join(folder_path, "Indexed_Database_Output")
        os.makedirs(indexed_folder, exist_ok=True)

        for j in range(k):
            points_in_class = dataset_array[final_labels == j]
            file_name = os.path.join(indexed_folder, f"clase{j+1}.txt")
            np.savetxt(file_name, points_in_class, fmt='%d', delimiter=',')
            
        print(f"Indexed Database generada en: {indexed_folder}")

        return self.centroids, converged_iter

    def segment_with_trained_model(self, image_path):
        if self.centroids is None:
            return None, None, None
            
        img = cv2.imread(image_path)
        if img is None:
            return None, None, None
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pixels = img.reshape((-1, 3)).astype(np.float32)
        
        # Clasificamos la imagen nueva basándonos en los centroides globales ya entrenados
        distances = np.linalg.norm(pixels[:, np.newaxis] - self.centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        segmented_pixels = self.centroids[labels].astype(np.uint8)
        segmented_img = segmented_pixels.reshape(img.shape)
        
        # Calcular porcentajes para la clasificación final
        counts = np.bincount(labels, minlength=len(self.centroids))
        total_pixels = len(labels)
        percentages = (counts / total_pixels) * 100
        
        return img, segmented_img, percentages