import numpy as np
import pytest

from ..src.cost_simulation import CostSimulator


@pytest.fixture
def cost_simulator_obj():
    return CostSimulator(10, 1, 1000, 0.15)


def test_simulate_yield_costs(cost_simulator_obj):
    np.random.seed(0)
    costs = cost_simulator_obj.simulate_yield_costs(40, 44)
    assert np.isclose(costs, [1.19, 2.714]).all()


def test_simulate_2_school_costs(cost_simulator_obj):
    np.random.seed(0)
    costs = cost_simulator_obj.simulate_2_school_costs(40, 60, 44, 67)
    assert np.isclose(costs, [[3.24, 6.094]]).all()


def test_optimize_2sch():
    cs = CostSimulator(10, 1, 1, 0.15)
    np.random.seed(0)
    cost_grid = cs.optimize_2sch(6, 6)
    assert cost_grid.shape == (2, 2)
    assert np.isclose(cost_grid, [[2.0, 1.0], [1.0, 1.0]]).all()


def test_heuristic_set_capacity(cost_simulator_obj):
    best_cap = cost_simulator_obj.heuristic_set_capacity(40)
    assert best_cap == 44


def test_compute_cdf_inflate_sep(cost_simulator_obj):
    cdf_val = cost_simulator_obj.compute_cdf_inflate_sep(40, 40)
    assert np.isclose(cdf_val, 0.9631577914158603)


def test_simulate_multi_school_costs():
    n = 7
    true_caps = np.full(n, 40)
    inf_caps = np.array([48, 49, 44, 46, 50, 47, 46])  # target diff: [1, 0, -3, -1, 3, -1, 0]
    co = cu = 1
    iters = 1
    prob = 0.15
    cs = CostSimulator(co, cu, iters, prob, true_caps)
    np.random.seed(0)
    cs.simulate_multi_school_costs(inf_caps)
    expected = np.array([[1., 0.],
       [0., 0.],
       [0., 3.],
       [0., 4.],
       [0., 1.],
       [0., 2.],
       [0., 2.]])
    assert np.isclose(cs.costs, expected).all()
