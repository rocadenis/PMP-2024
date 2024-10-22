
from pgmpy.models import MarkovNetwork
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import VariableElimination
import networkx as nx
import matplotlib.pyplot as plt

# a)
mm = MarkovNetwork()

mm.add_edges_from([('A1', 'A2'), ('A1', 'A3'), ('A2', 'A4'), ('A2', 'A5'), ('A3', 'A4'), ('A4', 'A5')])

plt.figure(figsize=(8,6))  
pos = nx.circular_layout(mm)  
nx.draw(mm, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10, font_weight='bold', edge_color='gray')
plt.show()

# b)
import numpy as np
import itertools

#definim coeficientii
t1, t2, t3, t4, t5 = 1, 2, 3, 4, 5 

states = list(itertools.product([0, 1], repeat=5))

probabilities = []
for state in states:
    A1, A2, A3, A4, A5 = state
    prob = np.exp(t1 * A1 + t2 * A2 + t3 * A3 + t4 * A4 + t5 * A5)
    probabilities.append((state, prob))

Z = sum(prob for _, prob in probabilities)

#normalizam probabilitatile
normalized_probabilities = [(state, prob / Z) for state, prob in probabilities]

#starea cu probabilitatea maxima
max_prob_state = max(normalized_probabilities, key=lambda x: x[1])

print(f"Starea cu probabilitatea maximă: {max_prob_state[0]} cu probabilitatea {max_prob_state[1]}")
