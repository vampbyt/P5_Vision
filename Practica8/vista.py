from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QFileDialog, QSpinBox, QMessageBox, QSizePolicy)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt

class KMeansView(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Práctica 7 - Segmentación (K-Means)")
        # Incrementamos el tamaño general de la ventana
        self.resize(1200, 700) 
        
        self.setStyleSheet("""
            QWidget {
                background-color: #000080;
                color: #FFFFFF;
                font-family: Arial;
                font-size: 14px;
            }
            QPushButton {
                background-color: #000040;
                border: 2px solid #4D4DFF;
                border-radius: 5px;
                padding: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1A1AFF;
            }
            QPushButton:disabled {
                background-color: #000020;
                border: 2px solid #333333;
                color: #888888;
            }
            QSpinBox {
                background-color: #000040;
                border: 1px solid #4D4DFF;
                padding: 5px;
                font-size: 16px;
            }
            QLabel {
                font-weight: bold;
            }
        """)

        main_layout = QVBoxLayout()
        control_layout = QHBoxLayout()
        
        self.btn_load = QPushButton("1. Cargar Dataset (Carpeta)")
        
        self.lbl_k = QLabel("¿K?:")
        self.spin_k = QSpinBox()
        self.spin_k.setRange(2, 20)
        self.spin_k.setValue(3)
        
        self.lbl_desc = QLabel("¿Descriptores?:")
        self.spin_desc = QSpinBox()
        self.spin_desc.setRange(100, 100000)
        self.spin_desc.setSingleStep(100)
        self.spin_desc.setValue(1000)
        
        self.btn_run = QPushButton("2. Entrenar K-Means")
        self.btn_run.setEnabled(False)

        self.btn_test = QPushButton("3. Probar Imagen")
        self.btn_test.setEnabled(False)

        # ... (debajo de self.btn_test = QPushButton("3. Probar Imagen")) ...
        
        self.btn_plot = QPushButton("4. Ver Distribución (3D)")
        self.btn_plot.setEnabled(False)

        control_layout.addWidget(self.btn_plot) # AGREGAMOS ESTO AL LAYOUT

        control_layout.addWidget(self.btn_load)
        control_layout.addWidget(self.lbl_k)
        control_layout.addWidget(self.spin_k)
        control_layout.addWidget(self.lbl_desc)
        control_layout.addWidget(self.spin_desc)
        control_layout.addWidget(self.btn_run)
        control_layout.addWidget(self.btn_test)
        
        images_layout = QHBoxLayout()
        
        self.lbl_img_original = QLabel("Imagen Original")
        self.lbl_img_original.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_img_original.setStyleSheet("border: 2px dashed #4D4DFF; background-color: #000030;")
        self.lbl_img_original.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        self.lbl_img_segmented = QLabel("Imagen Segmentada")
        self.lbl_img_segmented.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_img_segmented.setStyleSheet("border: 2px dashed #4D4DFF; background-color: #000030;")
        self.lbl_img_segmented.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        images_layout.addWidget(self.lbl_img_original)
        images_layout.addWidget(self.lbl_img_segmented)

        self.lbl_info = QLabel("Esperando dataset...")
        self.lbl_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_info.setStyleSheet("font-size: 18px; color: #00FF00; margin-top: 10px;")

        main_layout.addLayout(control_layout)
        # Le damos más proporción al área de imágenes (stretch)
        main_layout.addLayout(images_layout, stretch=1)
        main_layout.addWidget(self.lbl_info)
        
        self.setLayout(main_layout)

    def display_image(self, label, img_array):
        h, w, ch = img_array.shape
        bytes_per_line = ch * w
        q_img = QImage(img_array.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        
        # Ampliamos la escala de 400x400 a 550x550
        # Añadimos SmoothTransformation para que mantenga alta calidad al redimensionar
        label.setPixmap(pixmap.scaled(
            550, 550, 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        ))