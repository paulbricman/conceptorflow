from conceptors import *
import numpy as np
import matplotlib.pyplot as plt

epochs = 100
deltas = []
dims = range(3, 50)
zeros = []

for dim in dims:
    results = []
    results_conj = []
    zeros_here = []
    for i in range(epochs):
        states1 = np.random.rand(dim, 10)
        c1 = Conceptor().from_states(states1)

        states2 = np.random.rand(dim, 10)
        c2 = Conceptor().from_states(states2)

        c3 = conjunction([c1, c2])

        diff = c1.conceptor_matrix - c2.conceptor_matrix
        diff_conj = c1.conceptor_matrix - c3.conceptor_matrix

        evals = np.linalg.eigvals(diff).real
        evals_conj = np.linalg.eigvals(diff_conj).real
        zeros_here = len([e for e in evals_conj if abs(e) < 1e-8])
        
        results += [np.mean([e for e in evals if abs(e) > 1e-8])]
        results_conj += [np.mean([e for e in evals_conj if abs(e) > 1e-8])]

    deltas += [np.mean(results)]#[np.mean(results_conj) - np.mean(results)]
    zeros += [np.mean(zeros_here)]

#print(list(zip(dims, deltas)))
plt.plot(dims, deltas)
#plt.plot(dims, zeros)
plt.title('(conjunct-conjunction) by dimensionality (3-50) with 10 data points')
plt.show()
# plt.hist(results, bins=1000)
# plt.hist(results_conj, bins=1000)
# plt.title('distribution of eigenvalue medians of difference matrices (α=100)')
# plt.show()