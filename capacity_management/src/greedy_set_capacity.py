import numpy as np
from .cost_simulation import CostSimulator
from typing import Tuple


def greedy_set_capacity(cs: CostSimulator) -> Tuple[np.ndarray, float]:
    qhat = cs.true_caps.copy()
    n = len(qhat)

    cs.simulate_multi_school_costs(qhat)
    curr_cost = cs.costs.sum()

    while True:
        new_costs = np.empty(n)
        for i in range(n):
            cs.simulate_multi_school_costs(
                qhat + np.array([1 if j == i else 0 for j in range(n)])
            )
            new_costs[i] = cs.costs.sum()

        if (curr_cost - new_costs < 0).all():
            break

        i = np.argmax(curr_cost - new_costs)
        qhat[i] += 1
        curr_cost = new_costs[i]

    return qhat, curr_cost
