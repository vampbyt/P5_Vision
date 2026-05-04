import numpy as np
from abc import ABC, abstractmethod


class PuntoImagen:
    def __init__(self, vector_rgb, coordenadas):
        self.coordenadas = coordenadas
        self.rgb = np.array(vector_rgb)
        self.clasificado = False

class ClaseImagen:
    def __init__(self, nombre, pixeles_rgb, color):
        self.nombre = nombre
        self.pixeles = pixeles_rgb
        self.color = color

class _BasePerceptron:
    def __init__(self, ratio_de_aprendizaje=0.05, epocas=1000):
        self.lr = ratio_de_aprendizaje
        self.epocas = epocas
        self.minimos = None
        self.rangos = None

    def _ajustar_normalizacion(self, datos):
        self.minimos = np.min(datos, axis=0)
        maximos = np.max(datos, axis=0)
        self.rangos = maximos - self.minimos
        self.rangos[self.rangos == 0] = 1.0

    def _normalizar(self, datos):
        return (datos - self.minimos) / self.rangos

class PerceptronBinario(_BasePerceptron):
    def __init__(self, ratio_de_aprendizaje=0.05, epocas=1000):
        super().__init__(ratio_de_aprendizaje, epocas)
        self.pesos = None
        self.bias = 0.0
        self.clases_ordenadas = None

    def ajustar(self, datos, etiquetas, pesos_iniciales=None, bias_inicial=None):
        clases = np.unique(etiquetas)
        self.clases_ordenadas = clases
        
        self._ajustar_normalizacion(datos)
        x_norm = self._normalizar(datos)
        y_bin = np.where(etiquetas == clases[0], 0, 1)

        if pesos_iniciales is not None:
            self.pesos = np.array(pesos_iniciales, dtype=float)
        else:
            self.pesos = np.zeros(x_norm.shape[1], dtype=float)

        self.bias = float(bias_inicial) if bias_inicial is not None else 0.0

        for epoca in range(self.epocas):
            hubo_cambio = False
            for xi, yi in zip(x_norm, y_bin):
                suma_z = np.dot(xi, self.pesos) + self.bias
                salida = 1 if suma_z >= 0 else 0
                error = yi - salida
                
                if error != 0:
                    self.pesos += self.lr * error * xi
                    self.bias += self.lr * error
                    hubo_cambio = True
            if not hubo_cambio:
                print(f"Hubo convergencia en la época {epoca + 1}")
                return True
        print(f"No hubo convergencia tras {self.epocas} épocas.")
        return False

    def obtener_parametros(self):
        return self.pesos.copy(), float(self.bias), self.clases_ordenadas.copy()

class auxiliar_creacion_clases(ABC):
    @staticmethod
    def obtener_minimos_maximos_coordenadas(pixeles, distancia_diferencia=None, ancho_maximo=255, alto_maximo=255):
        x_min = min(pixel.coordenadas[0] for pixel in pixeles)
        x_max = max(pixel.coordenadas[0] for pixel in pixeles)
        y_min = min(pixel.coordenadas[1] for pixel in pixeles)
        y_max = max(pixel.coordenadas[1] for pixel in pixeles)
        if distancia_diferencia is not None:
            x_min = max(0, x_min - distancia_diferencia)
            x_max = min(ancho_maximo, x_max + distancia_diferencia)
            y_min = max(0, y_min - distancia_diferencia)
            y_max = min(alto_maximo, y_max + distancia_diferencia)
        return (x_min, y_min), (x_max, y_max)