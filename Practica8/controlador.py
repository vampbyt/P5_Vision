from PyQt6.QtWidgets import QFileDialog

class KMeansController:
    def __init__(self, modelo, vista):
        self.model = modelo
        self.view = vista
        self.dataset_path = None
        
        self.view.btn_load.clicked.connect(self.load_dataset)
        self.view.btn_run.clicked.connect(self.run_training)
        self.view.btn_test.clicked.connect(self.test_single_image)

    def load_dataset(self):
        folder_dir = QFileDialog.getExistingDirectory(self.view, "Seleccionar Carpeta del Dataset")
        if folder_dir:
            self.dataset_path = folder_dir
            self.view.lbl_img_original.setText(f"Dataset cargado:\n{folder_dir}")
            self.view.btn_run.setEnabled(True)
            self.view.lbl_info.setText("Listo para entrenar.")

    def run_training(self):
        k = self.view.spin_k.value()
        desc = self.view.spin_desc.value()
        
        self.view.lbl_info.setText("Entrenando... Revisa la terminal.")
        self.view.repaint()
        
        centroids, converged_iter = self.model.train_kmeans_on_dataset(self.dataset_path, k, desc)
        
        if centroids is not None:
            info_text = f"¡Entrenamiento Terminado! (Iteraciones: {converged_iter})"
            self.view.lbl_info.setText(info_text)
            
            print("--- Clases (Centroides) Encontradas ---")
            for i, c in enumerate(centroids):
                print(f"Clase C{i+1}: RGB({int(c[0])}, {int(c[1])}, {int(c[2])})")
            print(f"Iteración de convergencia: {converged_iter}")
            
            self.view.lbl_img_segmented.setText("Clasificador\nEntrenado")
            self.view.btn_test.setEnabled(True) 
        else:
            self.view.lbl_info.setText("Error: No se encontraron imágenes.")

    def test_single_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self.view, "Seleccionar Imagen a Segmentar", "", "Images (*.png *.jpg *.jpeg)")
        if file_name:
            original_img, segmented_img, percentages = self.model.segment_with_trained_model(file_name)
            
            if original_img is not None:
                self.view.display_image(self.view.lbl_img_original, original_img)
                self.view.display_image(self.view.lbl_img_segmented, segmented_img)
                
                # Armar el texto para la interfaz
                resultado_texto = "Clasificación de la imagen:\n"
                for i, pct in enumerate(percentages):
                    resultado_texto += f"C{i+1}: {pct:.2f}%  "
                
                self.view.lbl_info.setText(resultado_texto)
                
                # Imprimir en terminal
                print(f"\nReporte de clasificación para {file_name}:")
                for i, pct in enumerate(percentages):
                    print(f"Clase C{i+1}: {pct:.2f}% de la imagen")
                
                # =========================================================
                # Guardar el reporte en un archivo físico (.txt)
                # =========================================================
                import os
                if self.dataset_path:
                    report_path = os.path.join(self.dataset_path, "reporte_clasificacion.txt")
                    # Usamos "a" (append) para no borrar los reportes anteriores si pruebas varias imágenes
                    with open(report_path, "a", encoding="utf-8") as f:
                        f.write(f"--- Reporte de Prueba ---\n")
                        f.write(f"Imagen evaluada: {os.path.basename(file_name)}\n")
                        for i, pct in enumerate(percentages):
                            f.write(f"Clase C{i+1}: {pct:.2f}%\n")
                        f.write("\n")
                    
                    print(f"-> Reporte guardado con éxito en: {report_path}")