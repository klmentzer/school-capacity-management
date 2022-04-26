import numpy as np
from itertools import product

from .cost_simulation import CostSimulator


def grid_search_optimal_caps(co, cu, iters, p, true_caps):
    cs = CostSimulator(co, cu, iters, p, true_caps)
    int_lengths = true_caps * p / (1 - p)
    possible_caps = [np.arange(true_caps[i], true_caps[i] + int_lengths[i] + 1) for i in range(len(true_caps))]
    min_cost = np.inf
    best_capacities = None
    for caps in product(*possible_caps):
        inf_caps = np.array(caps).astype(int)
        cs.simulate_multi_school_costs(inf_caps)
        cost = cs.evaluation_metrics(inf_caps)['total_cost']
        if cost < min_cost:
            min_cost = cost
            best_capacities = inf_caps
    return best_capacities, min_cost
