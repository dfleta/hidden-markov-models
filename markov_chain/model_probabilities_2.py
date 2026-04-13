from pomegranate.markov_chain import MarkovChain
import torch

# Crear Markov chain
model = MarkovChain(k=1)

# En el ejercicio de clase
# Secuencia de observaciones: S S S S S R S S S R R
# 10 muestras

samples = [ [[0], [0]],
            [[0], [0]],
            [[0], [0]],
            [[0], [0]],
            [[0], [1]],
            [[1], [0]],
            [[0], [0]],
            [[0], [0]],
            [[0], [1]],
            [[1], [1]]]

X = torch.tensor(samples)
model_ejercicio = MarkovChain(k=1)
model_ejercicio.fit(X)

print("Ejercicio de clase, observaciones: S S S S S R S S S R R")
print("Probabilidades iniciales:")
# Probabilidades iniciales (Categorical)
print(model_ejercicio.distributions[0].probs[0])
# Probabilidades de transición o condicionadas (ConditionalCategorical)
print("Matriz de transicion:")
print(model_ejercicio.distributions[1].probs[0])

print("Probabilidad de la serie:")
print(model_ejercicio.probability(X))

'''
Probabilidades iniciales:
tensor([0.8000, 0.2000])
Matriz de transicion:
Parameter containing:
tensor([[0.7500, 0.2500],
        [0.5000, 0.5000]])
Probabilidad de la serie:
tensor([0.6000, 0.6000, 0.6000, 0.6000, 0.2000, 0.1000, 0.6000, 0.6000, 0.2000,
        0.1000])
'''