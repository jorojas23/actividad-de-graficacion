import numpy as np
import matplotlib.pyplot as plt

class LagrangeInterpolation:
    def __init__(self, dias, temperaturas):
        self.dias = dias
        self.temperaturas = temperaturas
        self.n = len(dias)

    def lagrange(self, x):
        result = 0
        for i in range(self.n):
            term = self.temperaturas[i]
            for j in range(self.n):
                if j != i:
                    term *= (x - self.dias[j]) / (self.dias[i] - self.dias[j])  # Use los valores reales de x
            result += term
        return result

    def plot(self):
        x_values = np.linspace(self.dias[0], self.dias[-1], 100)
        y_values = [self.lagrange(i) for i in x_values]
        
        plt.figure(figsize=(10, 6))
        plt.plot(x_values, y_values, label='Interpolación de Lagrange', color='blue')
        plt.scatter(self.dias, self.temperaturas, color='red', label='Datos Originales')
        plt.xticks(self.dias, ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo'])
        plt.title('Interpolación de Lagrange de Temperaturas')
        plt.xlabel('Días de la Semana')
        plt.ylabel('Temperaturas (°C)')
        plt.legend()
        plt.grid()
        plt.show()

dias = [0, 1, 2, 3, 4, 5, 6]  
temperaturas = [20, 22, 25, 24, 27, 29, 21]

interpolador = LagrangeInterpolation(dias, temperaturas)
interpolador.plot()