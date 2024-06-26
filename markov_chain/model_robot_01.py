from pomegranate.distributions import Categorical
from pomegranate.distributions import ConditionalCategorical
from pomegranate.markov_chain import MarkovChain
import torch

'''
La probabilidad de encender el robot aspirador y que se encuentre
en cada una de las habitaciones de la casa al comenzar el día es la misma.
Una vez encendido el robot, se mueve a una habitación contigua
o permanece en la misma con igual probabilidad.
'''

# Probabilidades iniciales
start = Categorical([[
    1/6,
    1/6,
    1/6,
    1/6,
    1/6,
    1/6
]])

# Modelo de transicion
transitions = ConditionalCategorical([[
    [1/3, 1/3, 0, 1/3, 0, 0],
    [1/2, 1/2, 0, 0, 0, 0],
    [0, 0, 1/2, 0, 0, 1/2],
    [1/2, 0, 0, 1/2, 0, 0],
    [0, 0, 0, 0, 1/2, 1/2],
    [0, 0, 1/3, 0, 1/3, 1/3]
]], [start])

# Crear Markov chain
model = MarkovChain([start, transitions])

print("Probabilidades iniciales:")
# Probabilidades iniciales (Categorical)
print(model.distributions[0].probs[0])
# Probabilidades de transición o condicionadas (ConditionalCategorical)
print("Matriz de transicion:")
print(model.distributions[1].probs[0])

# vector de probabilidades iniciales
v = model.distributions[0].probs[0]

# matriz de transicion
P = model.distributions[1].probs[0]

# Probabilidades en el segundo dia de la serie
w = torch.matmul(v, P)
print(f'w = {w}')

# Probabilidades en el tercer dia de la serie
u = torch.matmul(w, P)
print(f'u = {u}')

# Probabilidades en el cuarto dia de la serie
t = torch.matmul(u, P)
print(f't = {t}')

'''
w = tensor([0.2222, 0.1389, 0.1389, 0.1389, 0.1389, 0.2222])
u = tensor([0.2130, 0.1435, 0.1435, 0.1435, 0.1435, 0.2130])
t = tensor([0.2145, 0.1427, 0.1427, 0.1427, 0.1427, 0.2145])
'''
