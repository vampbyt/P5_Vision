import cv2
import numpy as np
import threading
import os
from gtts import gTTS
import pygame
import tkinter as tk
from tkinter import filedialog

class VistaVision:
    def __init__(self):
        self.window_name = "Sistema de Vision"

    # --- NUEVO: MENÚS Y EXPLORADOR DE ARCHIVOS ---
    def mostrar_menu_principal(self):
        print("\n" + "="*50)
        print(" " * 15 + "MENÚ PRINCIPAL")
        print("="*50)
        print("1. Modo Entrenamiento (Agregar a Base de Datos)")
        print("2. Modo Clasificación (Probar nueva imagen)")
        print("3. Salir del sistema")
        print("="*50)
        return input(">> Elige una opción (1, 2 o 3): ").strip()

    def seleccionar_imagen(self):
        """Abre un explorador de archivos gráfico para elegir la imagen cómodamente."""
        root = tk.Tk()
        root.withdraw() # Oculta la ventana principal negra de tkinter
        root.attributes('-topmost', True) # Fuerza que la ventana salga por encima de otras
        
        self.mostrar_mensaje("Abriendo explorador de archivos...")
        ruta_imagen = filedialog.askopenfilename(
            title="Selecciona una imagen para analizar",
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp"), ("Todos los archivos", "*.*")]
        )
        return ruta_imagen

    def preguntar_continuar(self):
        print("\n" + "-"*50)
        respuesta = input(">> ¿Deseas procesar otra imagen en este modo? (s/n): ").strip().lower()
        return respuesta == 's'

    # --- INTERFAZ COMÚN ---
    def mostrar_mensaje(self, msg):
        print(f"[Sistema] {msg}")

    def esperar_tecla(self):
        return cv2.waitKey(1) & 0xFF
        
    def cerrar_ventanas(self):
        cv2.destroyAllWindows()

    # --- INTERFAZ DE ENTRENAMIENTO ---
    def setup_ui_entrenamiento(self, initial_threshold, trackbar_callback, mouse_callback):
        cv2.namedWindow(self.window_name)
        cv2.createTrackbar("Umbral RGB", self.window_name, initial_threshold, 100, trackbar_callback)
        cv2.setMouseCallback(self.window_name, mouse_callback)

    def actualizar_display_entrenamiento(self, original_img, mask, seed=None):
        display = original_img.copy()
        if mask is not None:
            overlay = np.zeros_like(display)
            overlay[:] = (255, 255, 0) 
            roi = display[mask == 255]
            overlay_roi = overlay[mask == 255]
            display[mask == 255] = cv2.addWeighted(roi, 0.5, overlay_roi, 0.5, 0)
        
        if seed is not None:
            cv2.circle(display, seed, 4, (0, 0, 255), -1)
            
        cv2.imshow(self.window_name, display)

    def pedir_nombre_clase(self, clases_existentes):
        print("\n" + "="*50)
        print("Clases registradas actualmente:")
        print(f"👉 {', '.join(clases_existentes)}")
        print("-" * 50)
        print("Tip: Escribe el nombre tal cual aparece arriba para agregar más")
        print("experiencia a esa clase, o escribe uno nuevo para crearla.")
        return input(">> Ingresa el nombre de la clase: ").strip()

    def pedir_cantidad_descriptores(self):
        print("\n" + "="*50)
        while True:
            try:
                valor = input(">> ¿Cuántos descriptores deseas extraer de esta región? (Ej. 5406): ")
                return int(valor)
            except ValueError:
                print("Error: Por favor ingresa un número entero válido.")

    # --- INTERFAZ DE CLASIFICACIÓN ---
    def obtener_ventana_usuario(self, imagen):
        """Permite al usuario dibujar un ROI. Devuelve coordenadas y la imagen recortada."""
        roi_box = cv2.selectROI(self.window_name, imagen, fromCenter=False, showCrosshair=True)
        x, y, w, h = roi_box
        if w == 0 or h == 0:
            return None, None
        roi_img = imagen[y:y+h, x:x+w]
        return (x, y, w, h), roi_img

    def dibujar_resultado_clasificacion(self, imagen, coordenadas, clase):
        x, y, w, h = coordenadas
        cv2.rectangle(imagen, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(imagen, clase, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow(self.window_name, imagen)

    # --- AUDIO ---
    def reproducir_audio(self, texto):
        self.mostrar_mensaje(f"🔊 Audio: '{texto}'")
        
        def hablar():
            try:
                tts = gTTS(text=texto, lang='es')
                archivo_audio = "resultado_temporal.mp3"
                tts.save(archivo_audio)
                
                pygame.mixer.init()
                pygame.mixer.music.load(archivo_audio)
                pygame.mixer.music.play()
                
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
                    
                pygame.mixer.quit()
                os.remove(archivo_audio)
                
            except Exception as e:
                print(f"Error del sistema de audio: {e}")

        hilo_audio = threading.Thread(target=hablar)
        hilo_audio.start()