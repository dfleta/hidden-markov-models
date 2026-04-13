from pomegranate.distributions import Categorical
from pomegranate.distributions import ConditionalCategorical
from pomegranate.markov_chain import MarkovChain
import torch

'''
Encendemos el robot aspirador y lo colocamos en la habitación 1.
Una vez encendido el robot, se mueve a una habitación contigua
o permanece en la misma con igual probabilidad.
Ahora, las habitaciones 2 y 3, y 4 y 5 están conectadas.
Sin embargo, a la habitación 6 sólo puede accederse
desde la habitación 3.

+--------+--------+--------+-------+
|        |        |        |       |
|  H-1      H-2      H-3     H-6   |
|        |        |        |       |
+--    --+--------+--------+-------+
|        |
|  H-4   |
|        |
+--    --+
|        |
|  H-5   |
|        |
+--------+
'''

# Probabilidades iniciales
start = Categorical([[
    1.0,
    0,
    0,
    0,
    0,
    0
]])

# Modelo de transicion
transitions = ConditionalCategorical([[
    [1/3, 1/3, 0, 1/3, 0, 0],
    [1/3, 1/3, 1/3, 0, 0, 0],
    [0, 1/3, 1/3, 0, 0, 1/3],
    [1/3, 0, 0, 1/3, 1/3, 0],
    [0, 0, 0, 1/2, 1/2, 0],
    [0, 0, 1/3, 0, 0, 1/3]
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
w = tensor([0.3333, 0.3333, 0.0000, 0.3333, 0.0000, 0.0000])
u = tensor([0.3333, 0.2222, 0.1111, 0.2222, 0.1111, 0.0000])
t = tensor([0.2593, 0.2222, 0.1111, 0.2407, 0.1296, 0.0370])
'''
