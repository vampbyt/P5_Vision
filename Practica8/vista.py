from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QFileDialog, QSpinBox, QSizePolicy)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt

class KMeansView(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Práctica 7 - Detección por Segmentación (K-Means)")
        self.resize(1100, 700) 
        
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
                padding: 8px;
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
        
        self.btn_load = QPushButton("1. Cargar Dataset")
        
        self.lbl_k = QLabel("¿K?:")
        self.spin_k = QSpinBox()
        self.spin_k.setRange(2, 20)
        self.spin_k.setValue(3)
        
        self.lbl_desc = QLabel("Descriptores/Clase:")
        self.spin_desc = QSpinBox()
        self.spin_desc.setRange(100, 100000)
        self.spin_desc.setSingleStep(100)
        self.spin_desc.setValue(1000)
        
        self.btn_run = QPushButton("2. Entrenar K-Means")
        self.btn_run.setEnabled(False)

        # NUEVOS BOTONES SEPARADOS
        self.btn_load_test = QPushButton("3. Cargar Imagen Prueba")
        self.btn_load_test.setEnabled(False)
        
        self.btn_segment = QPushButton("4. Segmentar Puntos")
        self.btn_segment.setEnabled(False)

        self.btn_plot = QPushButton("5. Ver Gráfica (2D)")
        self.btn_plot.setEnabled(False)

        control_layout.addWidget(self.btn_load)
        control_layout.addWidget(self.lbl_k)
        control_layout.addWidget(self.spin_k)
        control_layout.addWidget(self.lbl_desc)
        control_layout.addWidget(self.spin_desc)
        control_layout.addWidget(self.btn_run)
        control_layout.addWidget(self.btn_load_test)
        control_layout.addWidget(self.btn_segment)
        control_layout.addWidget(self.btn_plot) 
        
        images_layout = QHBoxLayout()
        
        self.lbl_img_main = QLabel("Esperando Imagen de Prueba")
        self.lbl_img_main.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_img_main.setStyleSheet("border: 2px dashed #4D4DFF; background-color: #000030;")
        self.lbl_img_main.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        images_layout.addWidget(self.lbl_img_main)

        self.lbl_info = QLabel("Esperando dataset...")
        self.lbl_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_info.setStyleSheet("font-size: 18px; color: #00FF00; margin-top: 10px;")

        main_layout.addLayout(control_layout)
        main_layout.addLayout(images_layout, stretch=1)
        main_layout.addWidget(self.lbl_info)
        
        self.setLayout(main_layout)

    def display_image(self, label, img_array):
        h, w, ch = img_array.shape
        bytes_per_line = ch * w
        q_img = QImage(img_array.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        
        max_w = label.width() if label.width() > 100 else 800
        max_h = label.height() if label.height() > 100 else 500
        
        label.setPixmap(pixmap.scaled(
            max_w, max_h, 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        ))