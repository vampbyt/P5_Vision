import sqlite3
import numpy as np

class BaseDatosIndexada:
    def __init__(self, db_name="descriptores_escenas.db"):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self._crear_tablas()

    def _crear_tablas(self):
        # Tabla para los nombres de las clases (ej. Cielo, Arena, Bosque)
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS clases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nombre_clase TEXT UNIQUE NOT NULL
            )
        ''')
        
        # Tabla para los descriptores (Vectores RGB de los centroides)
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS descriptores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                clase_id INTEGER,
                r REAL,
                g REAL,
                b REAL,
                FOREIGN KEY (clase_id) REFERENCES clases (id)
            )
        ''')
        
        # --- CREACIÓN DE ÍNDICES ---
        # Esto es clave para que al clasificar una nueva imagen, 
        # la base de datos encuentre la información a gran velocidad.
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_clase ON descriptores (clase_id)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_rgb ON descriptores (r, g, b)')
        
        self.conn.commit()

    def guardar_descriptores(self, nombre_clase, descriptores):
        """Guarda una lista de centroides/vectores asociados a una clase."""
        # 1. Registrar la clase si no existe
        self.cursor.execute('''
            INSERT OR IGNORE INTO clases (nombre_clase) VALUES (?)
        ''', (nombre_clase,))
        self.conn.commit()
        
        # Obtener el ID de la clase
        self.cursor.execute('SELECT id FROM clases WHERE nombre_clase = ?', (nombre_clase,))
        clase_id = self.cursor.fetchone()[0]
        
        # 2. Preparar e insertar los vectores RGB
        # Asumimos que descriptores es un numpy array con forma (N, 3)
        datos_a_insertar = [(clase_id, float(c[0]), float(c[1]), float(c[2])) for c in descriptores]
        
        self.cursor.executemany('''
            INSERT INTO descriptores (clase_id, r, g, b)
            VALUES (?, ?, ?, ?)
        ''', datos_a_insertar)
        
        self.conn.commit()

    def cargar_todos_los_descriptores(self):
        """Devuelve un diccionario con las clases y sus respectivos vectores."""
        self.cursor.execute('''
            SELECT c.nombre_clase, d.r, d.g, d.b 
            FROM descriptores d
            JOIN clases c ON d.clase_id = c.id
        ''')
        resultados = self.cursor.fetchall()
        
        diccionario_clases = {}
        for fila in resultados:
            nombre = fila[0]
            vector = np.array([fila[1], fila[2], fila[3]])
            if nombre not in diccionario_clases:
                diccionario_clases[nombre] = []
            diccionario_clases[nombre].append(vector)
            
        return diccionario_clases

    def cerrar(self):
        self.conn.close()