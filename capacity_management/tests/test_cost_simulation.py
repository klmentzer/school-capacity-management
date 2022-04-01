import numpy as np

from ..src.cost_simulation import (
    simulate_yield_costs,
    simulate_2_school_costs,
    optimize_2sch,
    heuristic_set_capacity,
    compute_cdf_inflate_sep, simulate_multi_school_costs,
)


def test_simulate_yield_costs():
    np.random.seed(0)
    costs = simulate_yield_costs(40, 44, 10, 1, 1000, 0.15)
    assert np.isclose(costs, [1.19, 2.714]).all()


def test_simulate_2_school_costs():
    np.random.seed(0)
    costs = simulate_2_school_costs(40, 60, 44, 67, 10, 1, 1000, 0.15)
    assert np.isclose(costs, [[3.24, 6.094]]).all()


def test_optimize_2sch():
    np.random.seed(0)
    cost_grid = optimize_2sch(12, 12, 0.15, 1, 10, 1)
    assert cost_grid.shape == (2, 2)
    assert np.isclose(cost_grid, [[4.0, 3.0], [3.0, 4.0]]).all()


def test_heuristic_set_capacity():
    best_cap = heuristic_set_capacity(40, 10, 1, 0.15)
    assert best_cap == 44


def test_compute_cdf_inflate_sep():
    cdf_val = compute_cdf_inflate_sep(40, 40, 10, 1, 0.15)
    assert np.isclose(cdf_val, 0.9631577914158603)


def test_simulate_multi_school_costs():
    n = 7
    true_caps = np.full(n, 40)
    inf_caps = np.array([46, 49, 51, 48, 43, 49, 46])  # target diff: [1, 0, -3, -1, 3, -1, 0]
    co = cu = 1
    iters = 1
    prob = 0.15
    np.random.seed(0)
    costs = simulate_multi_school_costs(true_caps, inf_caps, co, cu, iters, prob)
    expected = np.array([[1., 0.],
       [0., 0.],
       [0., 3.],
       [0., 4.],
       [0., 1.],
       [0., 2.],
       [0., 2.]])
    assert np.isclose(costs, expected).all()
