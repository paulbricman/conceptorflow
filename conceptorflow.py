import numpy as np
from numpy.linalg import inv
from numpy import identity


class Conceptor:
    def from_states(self, state_cloud, aperture=0.1):
        self.aperture = aperture
        self.correlation_matrix = np.corrcoef(state_cloud)
        self.dims = len(self.correlation_matrix)

        # Equation (7), page 36
        self.conceptor_matrix = inv(
            (self.correlation_matrix + aperture ** (-2) * identity(self.dims))) * self.correlation_matrix

        return self

    def from_conceptor_matrix(self, conceptor_matrix, aperture=0.1):
        self.conceptor_matrix = conceptor_matrix
        self.dims = len(conceptor_matrix)
        self.aperture = None
        self.correlation_matrix = None

        return self


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
    result = inv(id + inv(x.conceptor_matrix * inv(id - x.conceptor_matrix) +
                 y.conceptor_matrix * inv(id - y.conceptor_matrix)))
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
    diff_y_x = y.conceptor_matrix - x.conceptor_matrix
    diff_x_y = x.conceptor_matrix - y.conceptor_matrix

    if is_pos_def(diff_x_y):
        return 1
    elif is_pos_def(diff_y_x):
        return -1
    else:
        return 0


def is_pos_def(A):
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False
