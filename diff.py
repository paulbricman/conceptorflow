from conceptors import *
import pickle
import numpy as np
import os


def diff(c1, c2):    
    diff = Conceptor().from_conceptor_matrix(c1.conceptor_matrix - c2.conceptor_matrix)
    evals = sorted(np.linalg.eigvals(diff.conceptor_matrix).real, reverse=True)
    #print(evals)
    
    if len([e for e in evals if e != 0]) == 0:
        pos_ratio = 1
        area_ratio = 1
    else:
        pos_ratio = len([e for e in evals if e > 0]) / len([e for e in evals if e != 0])
        area_ratio = sum([e for e in evals if e > 0]) / sum([np.abs(e) for e in evals if e != 0])

    return [np.mean(evals), np.median(evals), pos_ratio, area_ratio]


def load_all_conceptors(state_clouds_directory='./clouds'):
    concepts = {}

    for filename in os.listdir(state_clouds_directory):
        state_cloud = pickle.load(open(state_clouds_directory + '/' + filename, 'rb'))
        if isinstance(state_cloud, list) and len(state_cloud) > 768:
            print('(*) loading conceptor of:', filename)
            concepts[filename.split('.')[0]] = Conceptor().from_states(state_cloud)

    return concepts


def compute_adjacency_matrix():
    concepts = load_all_conceptors()
    diffs = np.zeros((len(concepts.keys()), len(concepts.keys()), 4))

    for c1_idx, c1 in enumerate(concepts.values()):
        for c2_idx, c2 in enumerate(concepts.values()):

            print(list(concepts.keys())[c1_idx], list(concepts.keys())[c2_idx])
            diff_metrics = diff(c1, c2)
            diffs[c1_idx][c2_idx] = diff_metrics

    return [list(concepts.keys()), diffs]


diffs = compute_adjacency_matrix()
print(diffs)
pickle.dump(diffs, open('diffs/diffs.pickle', 'wb+'))