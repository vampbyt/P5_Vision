from PyQt6.QtWidgets import QFileDialog
import matplotlib.pyplot as plt
import numpy as np
import os

class KMeansController:
    def __init__(self, modelo, vista):
        self.model = modelo
        self.view = vista
        self.dataset_path = None
        
        self.view.btn_load.clicked.connect(self.load_dataset)
        self.view.btn_run.clicked.connect(self.run_training)
        self.view.btn_test.clicked.connect(self.test_single_image)
        self.view.btn_plot.clicked.connect(self.plot_distribution)

    def load_dataset(self):
        folder_dir = QFileDialog.getExistingDirectory(self.view, "Seleccionar Carpeta del Dataset")
        if folder_dir:
            self.dataset_path = folder_dir
            self.view.lbl_img_main.setText(f"Dataset cargado:\n{folder_dir}")
            self.view.btn_run.setEnabled(True)
            self.view.lbl_info.setText("Listo para entrenar.")

    def run_training(self):
        k = self.view.spin_k.value()
        desc = self.view.spin_desc.value()
        
        self.view.lbl_info.setText("Entrenando... Revisa la terminal.")
        self.view.repaint()
        
        centroids, converged_iter = self.model.entrenamiento_kmeans_centro(self.dataset_path, k, desc)
        
        if centroids is not None:
            info_text = f"¡Entrenamiento Terminado! (Iteraciones: {converged_iter})"
            self.view.lbl_info.setText(info_text)
            
            print("--- Clases (Centroides) Encontradas ---")
            for i, c in enumerate(centroids):
                print(f"Clase C{i+1}: RGB({int(c[0])}, {int(c[1])}, {int(c[2])})")
            print(f"Iteración de convergencia: {converged_iter}")
            
            self.view.lbl_img_main.setText("Clasificador K-Means Entrenado.\nPresiona 'Probar Imagen'")
            self.view.btn_test.setEnabled(True) 
            self.view.btn_plot.setEnabled(True)
        else:
            self.view.lbl_info.setText("Error: No se encontraron imágenes.")

    def plot_distribution(self):
        if self.model.dataset_array is None or self.model.centroids is None:
            return

        data = self.model.dataset_array
        centroids = self.model.centroids

        if len(data) > 2000:
            indices = np.random.choice(len(data), 2000, replace=False)
            data = data[indices]

        fig, ax = plt.subplots(figsize=(9, 7))

        x_min, x_max = 0, 255
        y_min, y_max = 0, 255
        xx, yy = np.meshgrid(np.arange(x_min, x_max + 1, 2),
                             np.arange(y_min, y_max + 1, 2))
        
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        centroids_2d = centroids[:, :2]
        
        distances = np.linalg.norm(grid_points[:, np.newaxis] - centroids_2d, axis=2)
        Z = np.argmin(distances, axis=1)
        Z = Z.reshape(xx.shape)
        
        ax.contourf(xx, yy, Z, alpha=0.15, cmap='tab10')

        colors = data / 255.0 
        ax.scatter(data[:, 0], data[:, 1], c=colors, marker='o', alpha=0.6, edgecolors='none', label='Muestras')

        for i, (cx, cy) in enumerate(centroids_2d):
            ax.scatter(cx, cy, c='black', linewidth=2, marker='X', s=250, zorder=5)
            ax.text(cx, cy + 8, f'C{i+1}', fontsize=12, fontweight='bold', 
                    color='black', ha='center', va='bottom', 
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', pad=2), zorder=6)

        ax.set_title("Distribución de Muestras y Regiones de Competencia", fontsize=14, fontweight='bold')
        ax.set_xlabel('Intensidad de Rojo (R)')
        ax.set_ylabel('Intensidad de Verde (G)')
        
        ax.set_xlim(0, 255)
        ax.set_ylim(0, 255)
        ax.grid(True, linestyle='--', alpha=0.4)
        
        plt.show()

    def test_single_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self.view, "Seleccionar Imagen a Analizar", "", "Images (*.png *.jpg *.jpeg)")
        if file_name:
            # Ahora el modelo solo nos devuelve la imagen con cajas y los porcentajes
            img_result, percentages = self.model.segmentar_con_imagen_prueba(file_name)
            
            if img_result is not None:
                # Enviamos la imagen al único panel que tenemos
                self.view.display_image(self.view.lbl_img_main, img_result)
                
                resultado_texto = "Clasificación:\n"
                for i, pct in enumerate(percentages):
                    resultado_texto += f"C{i+1}: {pct:.2f}%  "
                
                self.view.lbl_info.setText(resultado_texto)
                
                if self.dataset_path:
                    report_path = os.path.join(self.dataset_path, "reporte_clasificacion.txt")
                    with open(report_path, "a", encoding="utf-8") as f:
                        f.write(f"--- Reporte de Prueba ---\n")
                        f.write(f"Imagen evaluada: {os.path.basename(file_name)}\n")
                        for i, pct in enumerate(percentages):
                            f.write(f"Clase C{i+1}: {pct:.2f}%\n")
                        f.write("\n")