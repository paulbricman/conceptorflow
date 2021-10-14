import numpy as np
from numpy.core.fromnumeric import sort
from numpy.linalg import inv, norm
from numpy import identity, dot
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt


class Conceptor:
    def from_states(self, state_cloud, aperture=10):
        self.aperture = aperture
        self.correlation_matrix = np.corrcoef(np.array(state_cloud))
        self.dims = len(self.correlation_matrix)

        # Equation (7), page 36
        self.conceptor_matrix = \
            inv(self.correlation_matrix + aperture ** (-2) * identity(self.dims)) @ self.correlation_matrix

        return self


    def from_conceptor_matrix(self, conceptor_matrix, aperture=0.1):
        self.conceptor_matrix = conceptor_matrix
        self.dims = len(conceptor_matrix)
        self.aperture = aperture
        self.correlation_matrix = None

        return self

    
    def plot_spectrum(self):
        u, s, vh = np.linalg.svd(self.conceptor_matrix)
        s = sorted(s, reverse=True)
        fig, ax = plt.subplots()
        ax.set_title('Singular values of conceptor matrix')
        ax.bar(list(range(self.dims, 0, -1)), s)
        ax.invert_xaxis()
        ax.plot()
        plt.show()


def binary_conjunction(x, y):
    # Equation (30), page 52
    result = inv(inv(x.conceptor_matrix) + inv(
        y.conceptor_matrix) - identity(x.dims))
    return Conceptor().from_conceptor_matrix(result)


def conjunction(conceptors):
    assert len(conceptors) >= 2

    result = binary_conjunction(conceptors[0], conceptors[1])

    for conceptor in conceptors[2:]:
        result = binary_conjunction(result, conceptor)

    return result


def binary_disjunction(x, y):
    # Equation (26), page 52
    id = identity(x.dims)
    result = inv(id + inv(x.conceptor_matrix @ inv(id - x.conceptor_matrix) +
                 y.conceptor_matrix @ inv(id - y.conceptor_matrix)))
    return Conceptor().from_conceptor_matrix(result)


def disjunction(conceptors):
    assert len(conceptors) >= 2

    result = binary_disjunction(conceptors[0], conceptors[1])

    for conceptor in conceptors[2:]:
        result = binary_disjunction(result, conceptor)

    return result


def negation(x):
    # Equation (28), page 52
    result = identity(x.dims) - x.conceptor_matrix
    return Conceptor().from_conceptor_matrix(result)


def compare(x, y):
    # Proposition 13, page 58
    diff_y_x = y.conceptor_matrix - x.conceptor_matrix
    diff_x_y = x.conceptor_matrix - y.conceptor_matrix

    if is_pos_def(diff_x_y):
        return 1
    elif is_pos_def(diff_y_x):
        return -1
    else:
        return 0


def aperture_adaptation(x, new_aperture):
    # Equation (16), page 44
    assert x.aperture
    aperture_ratio = new_aperture / x.aperture
    result = x.conceptor_matrix @ \
        inv(x.conceptor_matrix + aperture_ratio ** (-2)
            * (identity(x.dims) - x.conceptor_matrix))
    return Conceptor().from_conceptor_matrix(result, aperture=new_aperture)


def similarity(x, y):
    # Equation (10), page 38
    ui, si, vhi = np.linalg.svd(x.conceptor_matrix)
    uj, sj, vhj = np.linalg.svd(y.conceptor_matrix)
    result = (norm(sqrtm(np.diag(si)) @ ui.T @ uj @
              sqrtm(np.diag(sj))) ** 2) / (norm(si) * norm(sj))

    return result


def is_pos_def(x, pos_sv_tol=1e-16, flip_vecs_tol=1e-13):
    '''
    pos_sv_tol: tolerance for singular values to be considered positive
    flip_vecs_tol: tolerance for checking equality between U and V.T
    '''
    if not np.allclose(x, x.T):
        return False

    u, s, vh = np.linalg.svd(x)

    if np.all(s < -pos_sv_tol):
        return False

    for sv_idx, sv in enumerate(s):
        if sv > flip_vecs_tol:
            if not np.allclose(u.T[sv_idx], vh[sv_idx]):
                return False
        else:
            if not (np.allclose(u.T[sv_idx], vh[sv_idx]) or np.allclose(u.T[sv_idx], -vh[sv_idx])):
                return False
    
    return True
