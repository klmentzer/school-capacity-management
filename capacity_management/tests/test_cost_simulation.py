import pytest
import numpy as np

from ..src.cost_simulation import (
    simulate_yield_costs,
    simulate_2_school_costs,
    optimize_2sch,
    heuristic_set_capacity,
    compute_cdf_inflate_sep,
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
