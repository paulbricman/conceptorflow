from conceptors import *
import matplotlib.pyplot as plt
import pickle
import numpy as np

word1 = 'class'
word2 = 'instance'

c1 = Conceptor().from_conceptor_matrix(pickle.load(open('conceptors/' + word1 + '.pickle', 'rb')))
c2 = Conceptor().from_conceptor_matrix(pickle.load(open('conceptors/' + word2 + '.pickle', 'rb')))

diff = Conceptor().from_conceptor_matrix(c1.conceptor_matrix - c2.conceptor_matrix)
evals = sorted(np.linalg.eigvals(diff.conceptor_matrix).real, reverse=True)
plt.bar(range(diff.conceptor_matrix.shape[0]), evals)
plt.hlines(np.mean(evals), 0, 768, colors='red')
plt.title(word1 + ' - ' + word2)
plt.show()