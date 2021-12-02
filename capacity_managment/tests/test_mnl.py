import numpy as np

from ..src.mnl import MNL


def test_calculate_choice_probabilities():
    utilities = np.linspace(1, 6, 6)
    mnl = MNL(utilities)
    np.random.seed(1)
    probs = mnl.calculate_choice_probabilities()
    expected = [2.78527287e-05, 3.20646680e-05, 9.70908049e-01, 8.38513283e-04, 5.17034035e-03, 2.30231802e-02]
    assert np.isclose(probs, expected).all()


def test_sample_preference_ordering():
    utilities = np.linspace(1, 6, 6)
    mnl = MNL(utilities)
    np.random.seed(1)
    ranking = mnl.sample_preference_ordering()
    expected = np.array([2, 4, 5, 3, 1, 0])
    assert np.equal(ranking, expected).all()
