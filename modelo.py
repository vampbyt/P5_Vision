from itertools import product, combinations
from numpy import array, zeros, dot
import numpy as np

class Modelo:
    def __init__(self):
        self.X = np.array([
            [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1],
            [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 0], [1, 1, 1, 1]
        ])

        self.C_objetivo = np.array([0, 0, 1, 0, 1, 0, 1, 1])
        self.W = np.zeros(4)
        self.eta = 0.1 
        self.max_iter = 1000

    def parametros(self, eta, w_inicial):
        self.eta = eta
        self.W = np.array(w_inicial, dtype=float)

    def entrenar(self):
        iteracion = 0
        ha_convergido = False
        num_patrones = len(self.X)
        self.historial = []  # lista de cambios para mostrar en pantalla

        while not ha_convergido and iteracion < self.max_iter:
            iteracion += 1
            errores = 0
            cambios_iter = []  # cambios dentro de esta iteración

            for i in range(num_patrones):
                x = self.X[i]
                c = self.C_objetivo[i]

                red = np.dot(self.W, x)
                y = 1 if red >= 0 else 0

                if y != c:
                    errores += 1
                    if c == 1 and y == 0:
                        self.W = self.W + self.eta * x
                    else:
                        self.W = self.W - self.eta * x
                    cambios_iter.append({
                        'patron': i,
                        'A': int(x[1]), 'B': int(x[2]), 'C': int(x[3]),
                        'y': y, 'c': c,
                        'W': self.W.copy()
                    })

            self.historial.append({
                'iteracion': iteracion,
                'errores': errores,
                'cambios': cambios_iter
            })

            if errores == 0:
                ha_convergido = True

        return ha_convergido, iteracion, self.W