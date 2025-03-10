import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Clase para manejar los datos meteorológicos
class DatosMeteorologicos:
    def __init__(self, dias, temperaturas):
        self.dias = dias
        self.temperaturas = temperaturas
        self.dias_numericos = np.arange(len(dias))  # Convertir días a formato numérico para cálculos

    def obtener_dias(self):
        return self.dias_numericos

    def obtener_temperaturas(self):
        return self.temperaturas


# Clase para realizar la interpolación polinómica a trozos
class InterpolacionPolinomicaATrozos:
    def __init__(self, datos):
        self.datos = datos
        self.interpolador = None

    def realizar_interpolacion(self):
        # Crear el interpolador polinómico a trozos (cúbico)
        self.interpolador = interp1d(self.datos.obtener_dias(), self.datos.obtener_temperaturas(), kind='cubic')

    def evaluar_interpolacion(self, puntos):
        if self.interpolador is None:
            raise ValueError("Debe realizar la interpolación antes de evaluarla.")
        return self.interpolador(puntos)

    def graficar(self):
        # Datos originales
        dias_numericos = self.datos.obtener_dias()
        temperaturas = self.datos.obtener_temperaturas()

        # Interpolación
        puntos_finos = np.linspace(dias_numericos[0], dias_numericos[-1], 100)
        temperaturas_interpoladas = self.evaluar_interpolacion(puntos_finos)

        # Graficar
        plt.figure(figsize=(8, 5))
        plt.plot(dias_numericos, temperaturas, 'o', label='Datos Originales', color='red')
        plt.plot(puntos_finos, temperaturas_interpoladas, '-', label='Interpolación Polinómica a Trozos', color='blue')
        plt.xticks(dias_numericos, self.datos.dias)  # Etiquetas de los días
        plt.xlabel('Días')
        plt.ylabel('Temperatura (°C)')
        plt.title('Interpolación Polinómica a Trozos de las Temperaturas Diarias')
        plt.legend()
        plt.grid(True)
        plt.show()


# Datos proporcionados
dias = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
temperaturas = [20, 22, 25, 24, 27, 29, 21]

# Crear instancia de DatosMeteorologicos
datos = DatosMeteorologicos(dias, temperaturas)

# Crear instancia de InterpolacionPolinomicaATrozos y realizar interpolación
interpolacion = InterpolacionPolinomicaATrozos(datos)
interpolacion.realizar_interpolacion()
interpolacion.graficar()
