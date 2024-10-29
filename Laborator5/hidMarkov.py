import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm

states = ["dificil", "mediu", "usor"]
observations = [0, 1, 2, 3]  # FB=0, B=1, S=2, NS=3

state_probability = np.array([1/3, 1/3, 1/3])

transition_probability = np.array([
    [0, 0.5, 0.5],
    [0.5, 0.25, 0.25],
    [0.5, 0.25, 0.25]
])

emission_probability = np.array([
    [0.1, 0.2, 0.4, 0.3],
    [0.15, 0.25, 0.5, 0.1],
    [0.2, 0.3, 0.4, 0.1]
])

model = hmm.CategoricalHMM(n_components=len(states))
model.startprob_ = state_probability
model.transmat_ = transition_probability
model.emissionprob_ = emission_probability

obs_indices = np.array([0, 0, 2, 1, 1, 2, 1, 1, 3, 1, 1, 2]).reshape(-1, 1)

logprob, hidden_states = model.decode(obs_indices, algorithm="viterbi")

print("Cea mai probabilă secvență de dificultăți:", [states[i] for i in hidden_states])
print("Probabilitatea secvenței:", np.exp(logprob))

plt.plot(hidden_states, '-o', label="Dificultatea testului")
plt.xlabel('Pasul de timp')
plt.ylabel('Stare probabilă')
plt.title("Secvența de dificultăți a testelor")
plt.legend()
plt.show()