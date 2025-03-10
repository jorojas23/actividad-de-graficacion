import numpy as np
import matplotlib.pyplot as plt

class EstructuraPuente:
    def __init__(self, rigidez):
        self.rigidez = np.array(rigidez)

    def metodo_potencia(self, iteraciones=1000, tolerancia=1e-6):
        b_k = np.random.rand(self.rigidez.shape[1])
        autovalores = []
        for i in range(iteraciones):
            b_k1 = np.dot(self.rigidez, b_k)
            b_k1_norm = np.linalg.norm(b_k1)
            b_k = b_k1 / b_k1_norm
            autovalor = np.dot(b_k.T, np.dot(self.rigidez, b_k))
            autovalores.append(autovalor)
            if np.linalg.norm(b_k1 - b_k * b_k1_norm) < tolerancia:
                print(f"Método de Potencia convergió en {i + 1} iteraciones.")
                break
        return autovalor, b_k, autovalores

    def metodo_potencia_inverso(self, iteraciones=1000, tolerancia=1e-6):
        A_inv = np.linalg.inv(self.rigidez)
        b_k = np.random.rand(self.rigidez.shape[1])
        autovalores_inversos = []
        for i in range(iteraciones):
            b_k1 = np.dot(A_inv, b_k)
            b_k1_norm = np.linalg.norm(b_k1)
            b_k = b_k1 / b_k1_norm
            autovalor_inverso = np.dot(b_k.T, np.dot(A_inv, b_k))
            autovalores_inversos.append(1 / autovalor_inverso)
            if np.linalg.norm(b_k1 - b_k * b_k1_norm) < tolerancia:
                print(f"Método de Potencia Inverso convergió en {i + 1} iteraciones.")
                break
        return 1 / autovalor_inverso, b_k, autovalores_inversos

    def calcular_numero_condicion(self):
        autovalores = np.linalg.eigvals(self.rigidez)
        numero_condicion = np.max(autovalores) / np.min(autovalores)
        return numero_condicion

    def graficar_convergencia(self, autovalores, titulo, metodo):
        plt.figure(figsize=(10, 6))
        plt.plot(autovalores, marker='o', linestyle='-', color='b', label='Autovalor')
        plt.axhline(y=autovalores[-1], color='r', linestyle='--', label='Valor Final')
        plt.title(titulo)
        plt.xlabel("Iteraciones")
        plt.ylabel("Autovalor")
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    # Matriz de rigidez de ejemplo
    rigidez = [[4, -1, 0, -1],
               [-1, 4, -1, 0],
               [0, -1, 4, -1],
               [-1, 0, -1, 4]]

    puente = EstructuraPuente(rigidez)

    # Método de Potencia
    autovalor_dominante, vector_dominante, autovalores_potencia = puente.metodo_potencia()
    print(f"Autovalor Dominante: {autovalor_dominante}")
    print(f"Vector Propio Dominante: {vector_dominante}")
    puente.graficar_convergencia(autovalores_potencia, 
                                 "Convergencia del Autovalor Dominante (Método de Potencia)", 
                                 "Potencia")

    # Método de Potencia Inverso
    autovalor_pequeno, vector_pequeno, autovalores_inversos = puente.metodo_potencia_inverso()
    print(f"Autovalor Más Pequeño: {autovalor_pequeno}")
    print(f"Vector Propio Más Pequeño: {vector_pequeno}")
    puente.graficar_convergencia(autovalores_inversos, 
                                 "Convergencia del Autovalor Más Pequeño (Método de Potencia Inverso)", 
                                 "Potencia Inverso")

    # Número de Condición
    numero_condicion = puente.calcular_numero_condicion()
    print("Número de Condición:", numero_condicion)

    if numero_condicion > 30:  # Umbral típico para evaluar la estabilidad
        print("Advertencia: El diseño podría ser inestable.")
    else:
        print("El diseño parece estable.")

    # Interpretación de resultados
    print("\nComentario sobre vulnerabilidad estructural:")
    print("El autovalor más pequeño indica la dirección en la que la estructura es más vulnerable. "
          "Si este autovalor es bajo, significa que la estructura puede experimentar grandes "
          "deformaciones bajo ciertas cargas, lo que podría ser un punto de falla crítico. "
          "Se deben considerar refuerzos en estas direcciones para mejorar la integridad del puente.")