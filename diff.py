from conceptors import *
import pickle
import numpy as np

def load_conceptor(concept):
    return Conceptor().from_conceptor_matrix(pickle.load(open('conceptors/' + concept + '.pickle', 'rb')))


def diff(c1, c2):
    c1 = load_conceptor(c1)
    c2 = load_conceptor(c2)
    
    diff = Conceptor().from_conceptor_matrix(c1.conceptor_matrix - c2.conceptor_matrix)
    evals = sorted(np.linalg.eigvals(diff.conceptor_matrix).real, reverse=True)

    pos_ratio = len([e for e in evals if e > 0]) / len([e for e in evals if e != 0])
    area_ratio = sum([e for e in evals if e > 0]) / sum([np.abs(e) for e in evals if e != 0])

    return [np.mean(evals), np.median(evals), pos_ratio, area_ratio]

