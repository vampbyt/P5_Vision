import cv2
import numpy as np

class ControladorVision:
    def __init__(self, model, view):
        self.model = model
        self.view = view
        
    def iniciar(self):
        while True:
            opcion = self.view.mostrar_menu_principal()

            if opcion == '3':
                self.view.mostrar_mensaje("Saliendo del sistema. ¡Hasta luego!")
                break
                
            if opcion not in ['1', '2']:
                self.view.mostrar_mensaje("Error: Opción no válida. Intenta de nuevo.")
                continue

            procesando_imagenes = True
            while procesando_imagenes:
                ruta_imagen = self.view.seleccionar_imagen()
                
                if not ruta_imagen:
                    self.view.mostrar_mensaje("No se seleccionó ninguna imagen.")
                    break 
                
                try:
                    self.model.load_image(ruta_imagen)
                except FileNotFoundError as e:
                    self.view.mostrar_mensaje(str(e))
                    continue

                if opcion == '1':
                    self.ejecutar_entrenamiento()
                elif opcion == '2':
                    self.ejecutar_clasificacion()
                    
                procesando_imagenes = self.view.preguntar_continuar()

    def ejecutar_entrenamiento(self):
        self.model.limpiar_semillas() # Aseguramos que inicie limpio
        self.view.setup_ui_entrenamiento(
            initial_threshold=self.model.threshold,
            trackbar_callback=self.on_trackbar_change,
            mouse_callback=self.on_mouse_click
        )
        self.view.mostrar_mensaje("ENTRENAMIENTO: Haz múltiples clics izquierdo para agregar semillas.")
        self.view.mostrar_mensaje("Teclas: 'g' Guardar | 'r' Limpiar semillas | ESC Salir.")
        self.view.actualizar_display_entrenamiento(self.model.image, None)

        is_running = True
        while is_running:
            key = self.view.esperar_tecla()
            if key in [27, ord('q')]: 
                is_running = False
            elif key == ord('r'): # Limpiar pantalla
                self.model.limpiar_semillas()
                self.view.actualizar_display_entrenamiento(self.model.image, None)
                self.view.mostrar_mensaje("Semillas limpiadas.")
            elif key == ord('g'):
                if self.model.mask is not None and len(self.model.seeds) > 0:
                    n_desc = self.view.pedir_cantidad_descriptores()
                    vectores = self.model.get_feature_vectors(n_desc)
                    
                    if vectores is not None:
                        clases_actuales = self.model.obtener_nombres_clases()
                        nombre_clase = self.view.pedir_nombre_clase(clases_actuales)
                        self.model.guardar_en_bd(nombre_clase, vectores)
                        self.view.mostrar_mensaje(f"¡Éxito! Se inyectaron {len(vectores)} descriptores en '{nombre_clase}'.")
                        
                        # Limpiamos para el siguiente entrenamiento en la misma foto
                        self.model.limpiar_semillas()
                        self.view.actualizar_display_entrenamiento(self.model.image, None)
                    else:
                        self.view.mostrar_mensaje("Error: La región está vacía.")
                else:
                    self.view.mostrar_mensaje("Primero haz clic para generar regiones.")
        self.view.cerrar_ventanas()

    # (La función ejecutar_clasificacion() déjala exactamente como la tenías en la versión anterior que sí te funcionaba ventana por ventana)

    def on_trackbar_change(self, value):
        self.model.set_threshold(value)
        if len(self.model.seeds) > 0:
            self.model.procesar_region_growing()
            # Mostramos todas las semillas
            img_display = self.model.image.copy()
            for s in self.model.seeds:
                 cv2.circle(img_display, s, 4, (0, 0, 255), -1)
            self.view.actualizar_display_entrenamiento(img_display, self.model.mask)

    def on_mouse_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Clic Izquierdo = Agrega una nueva semilla y recalcula todo fusionando
            self.model.agregar_semilla(x, y)
            self.model.procesar_region_growing()
            
            img_display = self.model.image.copy()
            for s in self.model.seeds:
                 cv2.circle(img_display, s, 4, (0, 0, 255), -1)
            self.view.actualizar_display_entrenamiento(img_display, self.model.mask)
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Clic Derecho = Limpia las semillas por si te equivocas
            self.model.limpiar_semillas()
            self.view.actualizar_display_entrenamiento(self.model.image, None)
            self.view.mostrar_mensaje("Semillas limpiadas.")

    def ejecutar_clasificacion(self):
        self.model.cargar_bd_para_clasificar()
        self.view.mostrar_mensaje("Modo Clasificación. Dibuja ventanas y presiona ENTER. Presiona 'c' para finalizar.")
        
        imagen_dibujo = self.model.image.copy()
        vectores_ventanas = [] # Aquí guardaremos los colores crudos

        while True:
            coordenadas, roi_img = self.view.obtener_ventana_usuario(imagen_dibujo)
            
            if roi_img is None: 
                break
                
            # Calculamos el vector promedio de la ventana
            vector_promedio = np.mean(roi_img, axis=(0, 1))
            vectores_ventanas.append(vector_promedio)
            
            # Clasificación en vivo (solo visual) para que veas qué detectó temporalmente
            subclase_temp = self.model.clasificar_vector(vector_promedio)
            self.view.dibujar_resultado_clasificacion(imagen_dibujo, coordenadas, subclase_temp)

        self.view.cerrar_ventanas()
        
        if not vectores_ventanas:
            self.view.mostrar_mensaje("No seleccionaste ninguna ventana para clasificar.")
            return

        # --- AQUÍ OCURRE LA MAGIA: CLASIFICACIÓN ASCENDENTE JERÁRQUICA ---
        self.view.mostrar_mensaje("Fase 2: Aplicando Clasificación Ascendente Jerárquica...")
        vectores_fusionados = self.model.clasificacion_ascendente_jerarquica(vectores_ventanas, umbral_fusion=40.0)
        
        # Inferencia final sobre las regiones fusionadas
        subclases_detectadas = set()
        for vec in vectores_fusionados:
            subclase_final = self.model.clasificar_vector(vec)
            subclases_detectadas.add(subclase_final)
        
        escena = self.model.determinar_escena_final(subclases_detectadas)
        self.view.mostrar_mensaje(f"Regiones finales tras fusión: {list(subclases_detectadas)}")
        self.view.mostrar_mensaje(f"Escena Final: {escena}")
        
        self.view.reproducir_audio(f"La imagen analizada pertenece a la clase {escena}")
