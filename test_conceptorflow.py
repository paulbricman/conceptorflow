from numpy.core.numeric import identity
from conceptorflow import Conceptor, alignment, aperture_adaptation, compare, conjunction, disjunction, is_pos_def, negation, similarity
import conceptorflow as cf
import numpy as np
from numpy.linalg import inv, norm


def test_conceptor_pos_def():
    states = np.random.rand(2, 2)
    c = Conceptor().from_states(states)
    assert is_pos_def(c.conceptor_matrix)

    states = np.random.rand(5, 5)
    c = Conceptor().from_states(states)
    assert is_pos_def(c.conceptor_matrix)

    states = np.random.rand(10, 10)
    c = Conceptor().from_states(states)
    assert is_pos_def(c.conceptor_matrix)

def test_conjunction_commutative():
    states = np.random.rand(2, 2)
    c1 = Conceptor().from_states(states)

    states = np.random.rand(2, 2)
    c2 = Conceptor().from_states(states)

    assert np.allclose(conjunction([c1, c2]).conceptor_matrix, conjunction([c2, c1]).conceptor_matrix)

def test_conjunction_less_abstract():
    states = np.random.rand(2, 2)
    c1 = Conceptor().from_states(states)

    states = np.random.rand(2, 2)
    c2 = Conceptor().from_states(states)

    assert compare(conjunction([c1, c2]), c1) == -1
    assert compare(conjunction([c1, c2]), c2) == -1
    assert compare(c1, conjunction([c1, c2])) == 1
    assert compare(c2, conjunction([c1, c2])) == 1

def test_disjunction_commutative():
    states = np.random.rand(2, 2)
    c1 = Conceptor().from_states(states)

    states = np.random.rand(2, 2)
    c2 = Conceptor().from_states(states)

    assert np.allclose(disjunction([c1, c2]).conceptor_matrix, disjunction([c2, c1]).conceptor_matrix)

def test_disjunction_more_abstract():
    states = np.random.rand(2, 2)
    c1 = Conceptor().from_states(states)

    states = np.random.rand(2, 2)
    c2 = Conceptor().from_states(states)

    assert compare(disjunction([c1, c2]), c1) == 1
    assert compare(disjunction([c1, c2]), c2) == 1
    assert compare(c1, disjunction([c1, c2])) == -1
    assert compare(c2, disjunction([c1, c2])) == -1

def test_de_morgan():
    states = np.random.rand(2, 2)
    c1 = Conceptor().from_states(states)

    states = np.random.rand(2, 2)
    c2 = Conceptor().from_states(states)

    lhs = conjunction([c1, c2]).conceptor_matrix
    rhs = negation(disjunction([negation(c1), negation(c2)])).conceptor_matrix

    assert np.allclose(lhs, rhs)

def test_aperture_adaptation_reversible():
    states = np.random.rand(2, 2)
    c1 = Conceptor().from_states(states)

    c2 = aperture_adaptation(aperture_adaptation(c1, 1.2), 0.8)

    assert np.allclose(c1.conceptor_matrix, c2.conceptor_matrix)

def test_similarity():
    states = np.random.rand(3, 3)
    c1 = Conceptor().from_states(states)

    states = 10 * np.identity(3)
    c2 = Conceptor().from_states(states)

    assert np.isclose(similarity(c1, c1), 1)
    assert not np.isclose(similarity(c1, c2), 1)

def test_alignment():
    states = np.random.rand(3, 3)
    c1 = Conceptor().from_states(states)

    assert alignment(c1, states[0]) > alignment(c1, np.ones(3))

def test_correlation_matrix_recovery():
    states = np.random.rand(3, 3)
    true_correlation_matrix = np.corrcoef(states)
    c1 = Conceptor().from_states(states, 0.5)
    recovered_correlation_matrix = 0.5 ** (-2) * (c1.conceptor_matrix @ inv(identity(c1.dims) - c1.conceptor_matrix))

    print(true_correlation_matrix)
    print(recovered_correlation_matrix)

    assert np.allclose(true_correlation_matrix, recovered_correlation_matrix)