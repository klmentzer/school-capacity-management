import numpy as np
import pytest

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
    MNLCostSimulator(
        10, 1, 1, 0.15, np.array([44, 44]), "mnl", mus=np.array([5, 1])
    )  # runs test in instantiation


def test_validate_order_condition_error():
    with pytest.raises(ValueError) as e_info:
        MNLCostSimulator(
            10, 1, 1, 0.15, np.array([44, 44]), "mnl", mus=np.array([0, 1])
        )
    assert e_info.value.args[0] == "The provided mus do not satisfy the order condition."
