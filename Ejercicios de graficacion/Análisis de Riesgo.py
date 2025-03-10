import numpy as np
import matplotlib.pyplot as plt

class PortafolioRiesgo:
    def __init__(self, n_activos):
        self.n_activos = n_activos
        self.matriz_covarianza = self.generar_matriz_covarianza_controlada()
        
    def generar_matriz_covarianza_controlada(self):
        """Genera una matriz de covarianza controlada y positiva definida con autovalores diferenciados."""
        # Definimos los autovalores para la matriz (autovalor dominante más grande y el más pequeño)
        autovalores = np.array([1, 5, 10, 20, 100])  # Ejemplo de autovalores (menor a mayor riesgo)
        
        # Generamos una matriz diagonal con estos autovalores
        matriz_diag = np.diag(autovalores)
        
        # Para asegurar que la matriz sea simétrica y positiva definida, agregamos una pequeña perturbación
        # para controlar la relación entre activos.
        perturbacion = np.random.rand(self.n_activos, self.n_activos) * 0.1
        matriz_covarianza = matriz_diag + perturbacion + perturbacion.T
        
        # Garantizamos que la matriz sea positiva definida
        matriz_covarianza += np.eye(self.n_activos) * 1e-5  # Pequeña cantidad para asegurar positividad
        
        return matriz_covarianza

    def metodo_potencia(self, A, x0, tol=1e-6, max_iter=100):
        """Método de Potencia para encontrar el autovalor dominante (mayor riesgo)."""
        x = x0 / np.linalg.norm(x0)  # Normalizamos el vector inicial
        autovalores = []  # Lista para almacenar la convergencia de los autovalores

        for i in range(max_iter):
            y = np.dot(A, x)  # Multiplicación A * x
            autovalor = np.dot(x, y)   # Calculamos el autovalor aproximado
            autovalores.append(autovalor)
            x = y / np.linalg.norm(y)  # Normalizamos el vector

            # Criterio de convergencia
            if i > 0 and abs(autovalores[-1] - autovalores[-2]) < tol:
                break

        return autovalor, x, autovalores

    def metodo_potencia_inverso(self, A, x0, tol=1e-6, max_iter=100):
        """Método de Potencia Inverso para encontrar el autovalor más pequeño (menor riesgo)."""
        x = x0 / np.linalg.norm(x0)  # Normalizamos el vector inicial
        autovalores = []  # Lista para almacenar la convergencia de los autovalores

        for i in range(max_iter):
            y = np.linalg.solve(A, x)  # Resolvemos el sistema A * y = x
            autovalor = np.dot(x, y)   # Calculamos el autovalor aproximado
            autovalores.append(autovalor)
            x = y / np.linalg.norm(y)  # Normalizamos el vector

            # Criterio de convergencia
            if i > 0 and abs(autovalores[-1] - autovalores[-2]) < tol:
                break

        return autovalor, x, autovalores

    def numero_condicion(self, A):
        """Calcula el número de condición de la matriz A."""
        autovalores = np.linalg.eigvals(A)
        return np.max(autovalores) / np.min(autovalores)

    def analizar_riesgo(self):
        """Realiza el análisis de riesgo utilizando los métodos de potencia e inverso."""
        # Inicializamos un vector aleatorio
        x0 = np.random.rand(self.n_activos)

        # Método de potencia
        autovalor_dominante, autovector_dominante, autovalores_dominante = self.metodo_potencia(self.matriz_covarianza, x0)
        # Método de potencia inverso
        autovalor_pequeno, autovector_pequeno, autovalores_pequeno = self.metodo_potencia_inverso(self.matriz_covarianza, x0)

        # Número de condición
        num_condicion = self.numero_condicion(self.matriz_covarianza)

        # Mostrar resultados
        print(f"Autovalor dominante (mayor riesgo): {autovalor_dominante}")
        print(f"Autovector asociado (mayor riesgo): {autovector_dominante}")
        print(f"Autovalor más pequeño (menor riesgo): {autovalor_pequeno}")
        print(f"Autovector asociado (menor riesgo): {autovector_pequeno}")
        print(f"Número de Condición: {num_condicion}")

        # Graficar la convergencia
        plt.plot(autovalores_dominante, marker='o', linestyle='-', color='teal', label='Método de Potencia')
        plt.plot(autovalores_pequeno, marker='x', linestyle='--', color='orange', label='Método de Potencia Inverso')
        plt.title('Convergencia de Autovalores')
        plt.xlabel('Iteraciones')
        plt.ylabel('Autovalor')
        plt.legend()
        plt.grid(True)
        plt.show()

# Crear una instancia de la clase PortafolioRiesgo con 5 activos
portafolio = PortafolioRiesgo(n_activos=5)
portafolio.analizar_riesgo()
