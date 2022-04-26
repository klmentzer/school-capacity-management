import numpy as np
from ..src.optimal_policy_grid_search import grid_search_optimal_caps


def test_grid_search_optimal_caps():
    co = 10
    cu = 1
    iters = 1000
    p = 0.15
    true_caps = np.array([40] * 3)
    np.random.seed(0)
    actual_caps, actual_cost = grid_search_optimal_caps(co, cu, iters, p, true_caps)
    expected_caps = np.array([44, 46, 46])
    assert np.equal(actual_caps, expected_caps).all()
    assert np.isclose(15.133, actual_cost)
