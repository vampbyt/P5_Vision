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

        while not ha_convergido and iteracion < self.max_iter:
            iteracion += 1
            errores = 0

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
            
            if errores == 0:
                ha_convergido = True
        return ha_convergido, iteracion, self.W
    
