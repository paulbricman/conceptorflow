from conceptors import *
import pickle
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import networkx as nx


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


def eval_solution(mask, adjacency_matrix, heuristic='mean', print_components=False):
    prunes = 0
    vals = 0
    parentness = 0
    childness = 0
    elems = len(mask) * (len(mask) - 1)
    
    for row_idx, row in enumerate(mask):
        for col_idx, col in enumerate(row):
            if row_idx != col_idx:
                if col == True:
                    prunes -= 1
                    if heuristic == 'mean':
                        vals += adjacency_matrix[row_idx][col_idx][0]
                    elif heuristic == 'median':
                        vals += adjacency_matrix[row_idx][col_idx][1]
                    elif heuristic == 'pos_ratio':
                        vals += adjacency_matrix[row_idx][col_idx][2] - 0.5
                    elif heuristic == 'area_ratio':
                        vals += adjacency_matrix[row_idx][col_idx][3] - 0.5

    for col_idx, col in enumerate(np.transpose(mask)):
        parentness -= np.abs(max(0, len([row for row_idx, row in enumerate(col) if row == True and row_idx != col_idx]) - 1)) / len(mask)

    for row_idx, row in enumerate(mask):
        childness -= max(0, len([col for col_idx, col in enumerate(row) if col == True and col_idx != row_idx]) - 3) / len(mask)

    prunes = prunes / elems * 2
    vals = vals / elems * 1000
    parentness = parentness * 4
    childness = childness
    
    if print_components:
        print(*[round(e, 2) for e in [prunes, vals, parentness, childness]], sep='\t')

    return prunes + vals + parentness + childness


def simulated_annealing(weights, epochs=1000, heuristic='mean'):
    state = np.zeros((len(weights), len(weights)), dtype=bool)
    evals = []

    for epoch in tqdm(range(epochs)):
        temperature = (epochs - epoch) / epochs
        if temperature == 0:
            return state
        
        next = state.copy()
        next_row = np.random.randint(0, len(weights))
        next_col = np.random.randint(0, len(weights))
        next[next_row, next_col] = not next[next_row, next_col]

        next_eval = eval_solution(next, weights, heuristic)
        current_eval = eval_solution(state, weights, heuristic, print_components=True)
        evals += [current_eval]
        
        delta = next_eval - current_eval
        if delta > 0:
            state = next
        else:
            if np.random.uniform(0, 1) < (np.e ** (delta / temperature)):
                state = next

    plt.plot(evals)
    plt.show()

    return state


def show_graph_with_labels(adjacency_matrix, labels):
    for i in range(len(adjacency_matrix)):
        adjacency_matrix[i][i] = False
    rows, cols = np.where(adjacency_matrix == 1)

    # print(labels)
    # print(adjacency_matrix)

    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.DiGraph()
    gr.add_edges_from(edges)
    
    labels_dict = {}
    for node in gr.nodes:
        labels_dict[node] = labels[node]
    nx.draw(gr, node_size=500, labels=labels_dict, with_labels=True, connectionstyle='arc3, rad = 0.1')
    plt.show()


labels = pickle.load(open('diffs/diffs.pickle', 'rb'))[0][:]
diffs = np.array([e[:] for e in pickle.load(open('diffs/diffs.pickle', 'rb'))[1][:]])
toy_labels = ['fruit', 'juice', 'orange juice', 'apple', 'banana']
toy_label_indices = [labels.index(e) for e in toy_labels]
toy_diffs = [diffs[e] for e in toy_label_indices]
toy_diffs = [[row[e] for e in toy_label_indices] for row in toy_diffs]

toy_mask = simulated_annealing(toy_diffs, epochs=10000, heuristic='mean')
pickle.dump(toy_mask, open('diffs/mask.pickle', 'wb+'))

toy_mask = np.array([e[:] for e in pickle.load(open('diffs/mask.pickle', 'rb'))[:]])


eval_solution(toy_mask, toy_diffs, print_components=True, heuristic='mean')
show_graph_with_labels(toy_mask, toy_labels)