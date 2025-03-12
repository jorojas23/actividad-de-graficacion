import numpy as np
import matplotlib.pyplot as plt

class HermiteInterpolation:
    def __init__(self, dias, temperaturas):
        self.dias = dias
        self.temperaturas = temperaturas
        self.n = len(dias)
        self.q = np.zeros((self.n, self.n))
        
    def compute_q(self):
        for i in range(self.n):
            self.q[i][0] = self.temperaturas[i]
        
        for j in range(1, self.n):
            for i in range(self.n - j):
                self.q[i][j] = (self.q[i + 1][j - 1] - self.q[i][j - 1]) / (i + j - i)
    
    def hermite(self, x):
        result = 0
        for i in range(self.n):
            term = self.q[0][i]
            for j in range(i):
                term *= (x - j)  # usos j como el índice numérico
            result += term
        return result

    def plot(self):
        self.compute_q()
        
        x_values = np.linspace(0, self.n - 1, 100)
        y_values = [self.hermite(i) for i in x_values]
        
        plt.figure(figsize=(10, 6))
        plt.plot(x_values, y_values, label='Interpolación de Hermite', color='blue')
        plt.scatter(range(self.n), self.temperaturas, color='red', label='Datos Originales')
        plt.xticks(range(self.n), self.dias)
        plt.title('Interpolación de Hermite de Temperaturas')
        plt.xlabel('Días de la Semana')
        plt.ylabel('Temperaturas (°C)')
        plt.legend()
        plt.grid()
        plt.show()

dias = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
temperaturas = [20, 22, 25, 24, 27, 29, 21]

interpolador = HermiteInterpolation(dias, temperaturas)
interpolador.plot()
