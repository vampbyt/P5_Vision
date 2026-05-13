import numpy as np
from database_manager import BaseDatosIndexada

def inyectar_datos_sinteticos():
    db = BaseDatosIndexada()
    
    datos_entrenamiento = {
        'Cielo': [
            np.array([180.0, 110.0, 60.0]),    # Cielo azul intenso (Playa)
            np.array([230.0, 200.0, 140.0]),   # Cielo pálido/claro (Montaña)
            np.array([255.0, 240.0, 180.0]),   # Cielo casi blanco en el horizonte
            np.array([154.7, 115.8, 76.1])
        ],
        'Arena': [
            np.array([160.0, 200.0, 210.0]),   # Arena clara de tu foto de playa
            np.array([140.0, 180.0, 190.0]),   # Arena en la sombra
            np.array([147.8, 181.1, 171.3])
        ],
        'Agua': [
            # --- AGUA DE PLAYA ---
            np.array([130.0, 110.0, 40.0]),    # Océano azul marino profundo
            np.array([160.0, 140.0, 60.0]),    # Océano turquesa claro
            # --- AGUA DE RÍO/LAGO (Caso difícil) ---
            np.array([200.0, 180.0, 120.0]),   # Río reflejando cielo claro (Tu foto de montaña)
            np.array([180.0, 150.0, 100.0]),   # Agua de río con corriente
            np.array([150.0, 120.0, 80.0])     # Agua turbia clara
        ],
        'Nubes': [
            np.array([255.0, 255.0, 255.0]),
            np.array([230.0, 230.0, 230.0]),
            np.array([190.0, 190.0, 190.0])
        ],
        'Rocas': [
            np.array([130.0, 120.0, 120.0]),   # Montaña gris/púrpura de tu foto
            np.array([110.0, 100.0, 100.0]),   # Montaña en sombra
            np.array([150.0, 150.0, 140.0]),   # Piedras del río
            np.array([120.0, 120.0, 120.0])
        ],
        'Bosque': [
            # --- BOSQUE EN PRIMER PLANO (Vivo) ---
            np.array([70.0, 150.0, 100.0]),    # Hojas verde/amarillo (Frente de la montaña)
            np.array([50.0, 120.0, 60.0]),     # Verde estándar
            # --- BOSQUE DE FONDO (El caso que chocaba con el agua) ---
            np.array([70.0, 80.0, 50.0]),      # Pinos oscuros en la base de la montaña
            np.array([60.0, 90.0, 40.0]),      # Sombra profunda del bosque
            np.array([80.0, 100.0, 50.0]),     # Bosque lejano con atmósfera
            # --- EXTRA ---
            np.array([40.0, 90.0, 80.0])       # Palmera (verde muy oscuro y contrastado)
        ]
    }
    
    print("Inyectando valores de entrenamiento masivos en la base de datos...")
    
    # IMPORTANTE: Borramos los datos viejos para no tener duplicados inútiles
    #db.cursor.execute("DELETE FROM descriptores")
    #db.cursor.execute("DELETE FROM clases")
    #db.conn.commit()
    
    for clase, vectores in datos_entrenamiento.items():
        db.guardar_descriptores(clase, vectores)
        print(f" -> Guardados {len(vectores)} vectores para la clase '{clase}'")
        
    db.cerrar()
    print("\n¡Éxito! Base de datos lista.")

if __name__ == "__main__":
    inyectar_datos_sinteticos()