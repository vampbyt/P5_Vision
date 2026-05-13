import cv2

class ControladorVision:
    def __init__(self, model, view):
        self.model = model
        self.view = view
        
    def iniciar(self):
        while True:
            # 1. Mostrar el menú principal
            opcion = self.view.mostrar_menu_principal()

            if opcion == '3':
                self.view.mostrar_mensaje("Saliendo del sistema. ¡Hasta luego!")
                break
                
            if opcion not in ['1', '2']:
                self.view.mostrar_mensaje("Error: Opción no válida. Intenta de nuevo.")
                continue

            # 2. Ciclo para procesar múltiples imágenes sin volver al menú principal
            procesando_imagenes = True
            while procesando_imagenes:
                # Abrimos el explorador gráfico
                ruta_imagen = self.view.seleccionar_imagen()
                
                # Si el usuario cierra la ventana de selección sin elegir nada
                if not ruta_imagen:
                    self.view.mostrar_mensaje("No se seleccionó ninguna imagen.")
                    break 
                
                try:
                    self.model.load_image(ruta_imagen)
                except FileNotFoundError as e:
                    self.view.mostrar_mensaje(str(e))
                    continue

                # Ejecutamos la opción elegida
                if opcion == '1':
                    self.ejecutar_entrenamiento()
                elif opcion == '2':
                    self.ejecutar_clasificacion()
                    
                # 3. Preguntamos si quiere seguir en este modo
                procesando_imagenes = self.view.preguntar_continuar()

    def ejecutar_entrenamiento(self):
        self.view.setup_ui_entrenamiento(
            initial_threshold=self.model.threshold,
            trackbar_callback=self.on_trackbar_change,
            mouse_callback=self.on_mouse_click
        )
        self.view.mostrar_mensaje("Modo Entrenamiento. Clic para semilla. 'g' para guardar. ESC para salir.")
        self.view.actualizar_display_entrenamiento(self.model.image, None)

        is_running = True
        while is_running:
            key = self.view.esperar_tecla()
            if key in [27, ord('q')]: # ESC
                is_running = False
            elif key == ord('g'):
                if self.model.mask is not None:
                    # 1. Preguntamos la cantidad de descriptores
                    n_desc = self.view.pedir_cantidad_descriptores()
                    
                    # 2. Extraemos esa cantidad
                    vectores = self.model.get_feature_vectors(n_desc)
                    
                    if vectores is not None:
                        clases_actuales = self.model.obtener_nombres_clases()
                        nombre_clase = self.view.pedir_nombre_clase(clases_actuales)
                        
                        # 3. Guardamos la "nube de puntos" en la base de datos
                        self.model.guardar_en_bd(nombre_clase, vectores)
                        self.view.mostrar_mensaje(f"¡Éxito! Se inyectaron {len(vectores)} puntos descriptores en la clase '{nombre_clase}'.")
                    else:
                        self.view.mostrar_mensaje("Error: La región está vacía.")
                else:
                    self.view.mostrar_mensaje("Primero haz clic para generar una región.")
        self.view.cerrar_ventanas()

    def ejecutar_clasificacion(self):
        self.model.cargar_bd_para_clasificar()
        self.view.mostrar_mensaje("Modo Clasificación. Dibuja ventanas y presiona ENTER. Presiona 'c' para finalizar.")
        
        imagen_dibujo = self.model.image.copy()
        subclases_detectadas = set()

        while True:
            coordenadas, roi_img = self.view.obtener_ventana_usuario(imagen_dibujo)
            
            if roi_img is None: # Si el usuario presiona 'c' o cierra
                break
                
            # Mandamos el recorte al modelo para clasificar
            subclase = self.model.clasificar_ventana(roi_img)
            subclases_detectadas.add(subclase)
            
            # Mandamos a la vista que dibuje el resultado
            self.view.dibujar_resultado_clasificacion(imagen_dibujo, coordenadas, subclase)

        self.view.cerrar_ventanas()
        
        # Evaluar la escena final con las reglas
        escena = self.model.determinar_escena_final(subclases_detectadas)
        self.view.mostrar_mensaje(f"Subclases: {list(subclases_detectadas)}")
        self.view.mostrar_mensaje(f"Escena Final: {escena}")
        
        # Audio final
        self.view.reproducir_audio(f"La imagen analizada pertenece a la clase {escena}")

    # --- CALLBACKS DE ENTRENAMIENTO ---
    def on_trackbar_change(self, value):
        self.model.set_threshold(value)
        if self.model.seed is not None:
            self.model.procesar_region_growing()
            self.view.actualizar_display_entrenamiento(self.model.image, self.model.mask, self.model.seed)

    def on_mouse_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.model.set_seed(x, y)
            self.model.procesar_region_growing()
            self.view.actualizar_display_entrenamiento(self.model.image, self.model.mask, self.model.seed)