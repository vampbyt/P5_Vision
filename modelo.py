import numpy as np

class Modelo:
    def __init__(self):
        # Orden de la libreta: [A, B, C, Bias=1]
        self.X = np.array([
            # ---- CLASE 1 (C1) ----
            [0, 0, 0, 1], [1, 0, 1, 1], [1, 0, 0, 1], [1, 1, 0, 1],
            # ---- CLASE 2 (C2) ----
            [0, 0, 1, 1], [0, 1, 1, 1], [0, 1, 0, 1], [1, 1, 1, 1]
        ])
        
        # 0 representa C1, y 1 representa C2
        self.C_objetivo = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        self.W = np.zeros(4) 
        self.eta = 1.0
        self.historial = [] 

    def establecer_parametros(self, eta, w_inicial):
        self.eta = eta
        self.W = np.array(w_inicial, dtype=float)
        self.historial = [] # Limpiar historial en cada nuevo intento

    def entrenar(self):
        iteracion = 0
        ha_convergido = False
        num_patrones = len(self.X)
        paso_global = 1

        while not ha_convergido and iteracion < 1000:
            iteracion += 1
            errores = 0
            
            for i in range(num_patrones):
                x = self.X[i]
                c = self.C_objetivo[i]
                w_antes = self.W.copy()
                
                # Cálculo de Net (Producto punto)
                red = np.dot(self.W, x)
                sancion_str = "0"
                
                # REGLA ESTRICTA 
                if c == 0 and red >= 0:
                    errores += 1
                    self.W = self.W - (self.eta * x)
                    sancion_str = f"-X{paso_global}"
                elif c == 1 and red <= 0:
                    errores += 1
                    self.W = self.W + (self.eta * x)
                    sancion_str = f"+X{paso_global}"

                # Guardar registro para la tabla
                self.historial.append({
                    'paso': paso_global,
                    'x': x,
                    'w_antes': w_antes,
                    'net': red,
                    'sancion': sancion_str,
                    'w_nuevo': self.W.copy(),
                    'clase': "C1" if c == 0 else "C2"
                })
                paso_global += 1

            if errores == 0:
                ha_convergido = True

        return ha_convergido, iteracion