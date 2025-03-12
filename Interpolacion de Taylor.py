import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BarycentricInterpolator

class InterpolacionTaylor:
    def __init__(self, dias, temperaturas):
        self.dias = dias
        self.temperaturas = temperaturas
        self.n = len(dias)
        self.interpolador = BarycentricInterpolator(range(self.n), temperaturas)

    def taylor_series(self, x):
        return self.interpolador(x)

    def plot(self, x0_taylor=3):
        x_values = np.linspace(0, self.n - 1, 100)
        y_values_taylor = self.taylor_series(x_values)

        plt.figure(figsize=(12, 6))
        plt.plot(x_values, y_values_taylor, label='Interpolación de Taylor', color='green')
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
interpolador.plot()  # Interpolación de Taylor
