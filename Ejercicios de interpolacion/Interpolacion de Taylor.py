import numpy as np
import matplotlib.pyplot as plt
import math

class InterpolacionTaylor:
    def __init__(self, dias, temperaturas):
        self.dias = dias
        self.temperaturas = temperaturas
        self.n = len(dias)
    
    def derivada(self, n, x0):
        if n == 0:
            return self.temperaturas[x0]
        elif n == 1:
            if x0 == 0:
                return self.temperaturas[x0 + 1] - self.temperaturas[x0]
            elif x0 == self.n - 1:
                return self.temperaturas[x0] - self.temperaturas[x0 - 1]
            else:
                return (self.temperaturas[x0 + 1] - self.temperaturas[x0 - 1]) / 2
        elif n == 2:
            if x0 == 0 or x0 == self.n - 1:
                return 0
            else:
                return self.temperaturas[x0 + 1] - 2 * self.temperaturas[x0] + self.temperaturas[x0 - 1]
        else:
            return 0

    def taylor_series(self, x, x0, grado):

        resultado = 0
        for i in range(grado + 1):
            termino = (self.derivada(i, x0) / math.factorial(i)) * (x - x0) ** i
            resultado += termino
        return resultado

    def plot(self, x0_taylor=3, grado_taylor=3):

        x_values = np.linspace(0, self.n - 1, 100)
        y_values_taylor = [self.taylor_series(i, x0_taylor, grado_taylor) for i in x_values]

        plt.figure(figsize=(12, 6))
        plt.plot(x_values, y_values_taylor, label=f'Interpolación de Taylor (grado={grado_taylor})', color='green')
        plt.scatter(range(self.n), self.temperaturas, color='red', label='Datos Originales')
        plt.xticks(range(self.n), self.dias)
        plt.title('Interpolación de Taylor de Temperaturas')
        plt.xlabel('Días de la Semana')
        plt.ylabel('Temperaturas (°C)')
        plt.legend()
        plt.grid()
        plt.show()

# Datos
dias = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
temperaturas = [20, 22, 25, 24, 27, 29, 21]

# Crear una instancia de la clase
interpolador = InterpolacionTaylor(dias, temperaturas)

# Graficar la interpolación
interpolador.plot(x0_taylor=3, grado_taylor=3)  # Interpolación de Taylor alrededor del Jueves (índice 3)
