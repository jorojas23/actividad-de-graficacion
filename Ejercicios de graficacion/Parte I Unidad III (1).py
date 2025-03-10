import numpy as np
import random
import matplotlib.pyplot as plt

# Clase para generar matrices y calcular su número de condición
class Matriz:
    def __init__(self, n):
        self.matriz = np.random.uniform(-10, 10, size=(n, n))  # Generar matriz aleatoria
        self.condicion = np.linalg.cond(self.matriz)  # Calcular número de condición
        self.tamaño = len(self.matriz)

class GeneradorMatrices:
    def __init__(self):
        self.lista_matriz = []
        self.condiciones = []
        self.n = []
        self.llenar()
    
    def llenar(self):
        tamaño_inicial = 3  # Iniciar con matrices de tamaño 3x3
        cantidad_matrices = random.randint(5, 10)  # Generar entre 5 y 10 matrices
        for i in range(cantidad_matrices):
            matriz = Matriz(tamaño_inicial)
            self.lista_matriz.append(matriz)
            self.condiciones.append(matriz.condicion)
            self.n.append(matriz.tamaño)
            tamaño_inicial += 1  # Aumentar el tamaño de la matriz en cada iteración

# Visualización del número de condición en función del tamaño de la matriz
class VisualizarCondiciones:
    def __init__(self):
        self.matrices = GeneradorMatrices()
        self.graficar()
    
    def graficar(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.matrices.n, self.matrices.condiciones, marker="o", color="blue")
        plt.title("Número de Condición en Función del Tamaño de la Matriz")
        plt.xlabel("Tamaño de la Matriz")
        plt.ylabel("Número de Condición")
        plt.grid()
        plt.show()

# Implementación del Método de Potencia y Visualización de Convergencia
class MetodoPotencia:
    def __init__(self, matriz, tolerancia=1e-6, max_iter=1000):
        self.matriz = matriz
        self.tolerancia = tolerancia
        self.max_iter = max_iter
        self.autovalor = None
        self.autovector = None
        self.convergencia_autovalor = []
        self.convergencia_autovector = []

    def calcular(self):
        n = self.matriz.shape[0]
        v = np.random.rand(n)
        v = v / np.linalg.norm(v)  # Normalizar vector inicial

        for i in range(self.max_iter):
            # Multiplicación por la matriz
            w = np.dot(self.matriz, v)
            autovalor_aproximado = np.dot(w, v)
            self.convergencia_autovalor.append(autovalor_aproximado)

            # Normalizar el vector para la próxima iteración
            v = w / np.linalg.norm(w)
            self.convergencia_autovector.append(v)

            # Comprobar convergencia
            if self.autovalor is not None and abs(autovalor_aproximado - self.autovalor) < self.tolerancia:
                break

            self.autovalor = autovalor_aproximado

        self.autovector = v

    def graficar_convergencia(self):
        plt.figure(figsize=(10, 6))

        # Graficar convergencia del autovalor
        plt.subplot(2, 1, 1)
        plt.plot(self.convergencia_autovalor, marker="o", color="green")
        plt.title("Convergencia del Autovalor Dominante")
        plt.xlabel("Iteraciones")
        plt.ylabel("Autovalor Aproximado")
        plt.grid()

        # Graficar convergencia de las componentes del autovector dominante
        plt.subplot(2, 1, 2)
        autovector_transpuesto = np.array(self.convergencia_autovector).T
        for idx, componente in enumerate(autovector_transpuesto):
            plt.plot(componente, label=f"Componente {idx+1}")
        plt.title("Convergencia del Autovector Dominante")
        plt.xlabel("Iteraciones")
        plt.ylabel("Valores de las Componentes")
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.show()

# Clase para el Método de Potencia Inversa
class MetodoPotenciaInversa:
    def __init__(self, matriz, tolerancia=1e-6, max_iter=1000):
        self.matriz = matriz
        self.tolerancia = tolerancia
        self.max_iter = max_iter
        self.autovalor = None
        self.autovector = None
        self.convergencia = []

    def calcular(self):
        n = self.matriz.shape[0]
        v = np.random.rand(n)
        v = v / np.linalg.norm(v)

        for _ in range(self.max_iter):
            w = np.linalg.solve(self.matriz, v)  # Resolver sistema lineal (matriz * w = v)
            autovalor = np.dot(w, v)
            self.convergencia.append(1 / autovalor)  # Autovalor más pequeño es el inverso del autovalor dominante
            v = w / np.linalg.norm(w)

            if self.autovalor is not None and abs(1 / autovalor - self.autovalor) < self.tolerancia:
                break

            self.autovalor = 1 / autovalor

        self.autovector = v

    def graficar_convergencia(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.convergencia, marker='o', color='purple')
        plt.title('Convergencia del Método de Potencia Inversa (Autovalor Más Pequeño)')
        plt.xlabel('Iteraciones')
        plt.ylabel('Autovalor Más Pequeño Aproximado')
        plt.grid()
        plt.show()


if __name__ == "__main__":
    # Parte 1: Número de condición en función del tamaño de la matriz
    visualizar = VisualizarCondiciones()

    # Generar una matriz aleatoria para los métodos de potencia
    tamaño_matriz = 5
    matriz = np.random.uniform(-10, 10, size=(tamaño_matriz, tamaño_matriz))
    matriz = np.dot(matriz, matriz.T)  # Aseguramos que sea simétrica y positiva definida

    # Mostrar información básica de la matriz
    print("\nPropiedades de la Matriz Generada:")
    print("Tamaño:", tamaño_matriz, "x", tamaño_matriz)
    condicion_matriz = np.linalg.cond(matriz)
    print("Número de Condición:", condicion_matriz)

    # Método de Potencia
    metodo_potencia = MetodoPotencia(matriz)
    metodo_potencia.calcular()
    print("\nMétodo de Potencia")
    print("Autovalor Dominante:", metodo_potencia.autovalor)
    print("Autovector Dominante:", metodo_potencia.autovector)
    print("Iteraciones para Convergencia:", len(metodo_potencia.convergencia_autovalor))
    metodo_potencia.graficar_convergencia()

    # Método de Potencia Inversa
    metodo_potencia_inversa = MetodoPotenciaInversa(matriz)
    metodo_potencia_inversa.calcular()
    print("\nMétodo de Potencia Inversa")
    print("Autovalor Más Pequeño:", metodo_potencia_inversa.autovalor)
    print("Autovector del Autovalor Más Pequeño:", metodo_potencia_inversa.autovector)
    print("Iteraciones para Convergencia:", len(metodo_potencia_inversa.convergencia))
    metodo_potencia_inversa.graficar_convergencia()

    # Mostrar todos los autovalores reales de la matriz como referencia
    autovalores_reales = np.linalg.eigvals(matriz)
    print("\nAutovalores Reales de la Matriz:", autovalores_reales)

    # Gráfica Comparativa de Convergencia
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(len(metodo_potencia.convergencia_autovalor)),
        metodo_potencia.convergencia_autovalor,
        label="Autovalor Dominante (Método de Potencia)",
        color="green",
        marker="o",
    )
    plt.plot(
        range(len(metodo_potencia_inversa.convergencia)),
        metodo_potencia_inversa.convergencia,
        label="Autovalor Más Pequeño (Método de Potencia Inversa)",
        color="purple",
        marker="x",
    )
    plt.title("Comparación de Convergencia: Autovalor Dominante vs. Más Pequeño")
    plt.xlabel("Iteraciones")
    plt.ylabel("Autovalores Aproximados")
    plt.legend()
    plt.grid()
    plt.show()

