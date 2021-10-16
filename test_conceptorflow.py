from numpy.core.numeric import identity
from conceptorflow import Conceptor, aperture_adaptation, compare, conjunction, disjunction, is_pos_def, negation, similarity
import conceptorflow as cf
import numpy as np
from numpy.linalg import inv, norm, det


def test_conceptor_pos_def():
    states = np.random.rand(3, 3)
    c = Conceptor().from_states(states)

    u, s, vh = np.linalg.svd(c.conceptor_matrix)
    print(s)
    print(u)
    print(vh.T)

    assert is_pos_def(c.conceptor_matrix)

    states = np.random.rand(5, 5)
    c = Conceptor().from_states(states)
    assert is_pos_def(c.conceptor_matrix)

    states = np.random.rand(10, 10)
    c = Conceptor().from_states(states)
    assert is_pos_def(c.conceptor_matrix)


def test_conjunction_commutative():
    states = np.random.rand(3, 3)
    c1 = Conceptor().from_states(states)

    states = np.random.rand(3, 3)
    c2 = Conceptor().from_states(states)

    assert np.allclose(conjunction(
        [c1, c2]).conceptor_matrix, conjunction([c2, c1]).conceptor_matrix)


def test_conjunction_less_abstract():
    states1 = np.random.rand(5, 5)
    c1 = Conceptor().from_states(states1)

    states2 = np.random.rand(5, 5)
    c2 = Conceptor().from_states(states2)

    assert compare(conjunction([c1, c2]), c1) != 1
    assert compare(conjunction([c1, c2]), c2) != 1
    assert compare(c1, conjunction([c1, c2])) != -1
    assert compare(c2, conjunction([c1, c2])) != -1


def test_disjunction_commutative():
    states = np.random.rand(3, 3)
    c1 = Conceptor().from_states(states)

    states = np.random.rand(3, 3)
    c2 = Conceptor().from_states(states)

    assert np.allclose(disjunction(
        [c1, c2]).conceptor_matrix, disjunction([c2, c1]).conceptor_matrix)


def test_disjunction_more_abstract():
    states1 = np.random.rand(3, 3)
    c1 = Conceptor().from_states(states1)

    states2 = np.random.rand(3, 3)
    c2 = Conceptor().from_states(states2)

    assert compare(disjunction([c1, c2]), c1) == 1
    assert compare(disjunction([c1, c2]), c2) == 1
    assert compare(c1, disjunction([c1, c2])) == -1
    assert compare(c2, disjunction([c1, c2])) == -1


def test_de_morgan():
    states = np.random.rand(3, 3)
    c1 = Conceptor().from_states(states)

    states = np.random.rand(3, 3)
    c2 = Conceptor().from_states(states)

    print(c1.conceptor_matrix)
    print(c2.conceptor_matrix)
    print('---')
    print(det(c1.conceptor_matrix), det(c2.conceptor_matrix))
    stuff = inv(c1.conceptor_matrix) + inv(c2.conceptor_matrix) - identity(c1.dims)
    print(stuff, det(stuff))

    lhs = conjunction([c1, c2]).conceptor_matrix
    rhs = negation(disjunction([negation(c1), negation(c2)])).conceptor_matrix

    print(lhs)
    print(rhs)

    assert np.allclose(lhs, rhs, rtol=1e-1)


def test_aperture_adaptation_reversible():
    states = np.random.rand(3, 3)
    c1 = Conceptor().from_states(states)

    c2 = aperture_adaptation(aperture_adaptation(c1, 1.2), 10)

    assert np.allclose(c1.conceptor_matrix, c2.conceptor_matrix)


def test_similarity():
    states = np.random.rand(3, 3)
    c1 = Conceptor().from_states(states)

    states = 10 * np.identity(3)
    c2 = Conceptor().from_states(states)

    assert np.isclose(similarity(c1, c1), 1)
    assert not np.isclose(similarity(c1, c2), 1)


def test_correlation_matrix_recovery():
    states = np.random.rand(3, 3)
    true_correlation_matrix = np.corrcoef(states)
    c1 = Conceptor().from_states(states, 0.5)
    recovered_correlation_matrix = 0.5 ** (-2) * (
        c1.conceptor_matrix @ inv(identity(c1.dims) - c1.conceptor_matrix))

    assert np.allclose(true_correlation_matrix, recovered_correlation_matrix)


def test_plot_spectrum():
    states = np.random.rand(10, 10)
    c1 = Conceptor().from_states(states)
    c1.plot_spectrum()

    states = np.identity(10)
    c1 = Conceptor().from_states(states)
    c1.plot_spectrum()

    states = np.array([
        [100, 0, 0, 0, 0],
        [50, 0, 0, 0, 0],
        [0, 1, 0.0001, 0, 0],
        [0, 1, 0, 0.0001, 0],
        [-100, 0, 0, 0, 0.0001]])
    c1 = Conceptor().from_states(states)
    c1.plot_spectrum()
    assert True


def test_plot_ellipses():
    states = np.array([[1, 2, 3], [3, 4, 8.9]])
    c1 = Conceptor().from_states(states)

    states = np.array([[1, 1.7, 2], [-10, 2, 9]])
    c2 = Conceptor().from_states(states)

    c3 = cf.disjunction([c1, c2])
    c4 = cf.conjunction([c1, c2])

    cf.plot_ellipses([c1, c2, c3, c4])