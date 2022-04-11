import numpy as np
from ..src.mnl_cost_simulation import MNLCostSimulator


def test_calc_mnl_coefficients():
    mcs = MNLCostSimulator(
        10, 1, 1, 0.15, np.array([4, 4, 4]), "chain", mus=np.array([3, 2, 1])
    )
    actual = mcs.calc_mnl_coefficients()
    expected = np.array(
        [[1.0, 0.24472847, 0.1558482], [0.0, 1.0, 0.35897199], [0.0, 0.0, 1.0]]
    )
    assert np.isclose(actual, expected).all()

def test_validate_order_condition():
    mcs = MNLCostSimulator(
        10, 1, 1, 0.15, np.array([44, 44]), "chain", mus=np.array([5, 1])
    )
    mcs.validate_order_condition()