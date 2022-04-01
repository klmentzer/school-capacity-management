import numpy as np
from scipy.stats import binom


def simulate_yield_costs(
    true_cap: int, inf_cap: int, co: float, cu: float, iters: int, prob: float
) -> np.ndarray:
    """
    Function that estimates the true cost corresponding to the
    first school for a given true capacity and inflated capacity.
    Returns the cost as an array with the first entry as the
    overage and the second one as the underage cost.

    :param true_cap: true capacity of the school
    :param inf_cap: inflated capacity
    :param co: per unit cost of overage
    :param cu: per unit cost of underage
    :param iters: number of iterations
    :param prob: probability of dropout
    :return: len 2 array where first entry is overage cost and second is underage for the one school
    """
    yld = np.full(iters, inf_cap, dtype=float) - np.random.binomial(
        inf_cap, prob, size=iters
    )
    diff = yld - np.full(iters, true_cap, dtype=float)
    overage = co * sum(np.where(diff > 0, diff, 0)) / iters
    underage = cu * sum(np.where(-1 * diff > 0, -1 * diff, 0)) / iters
    costs = np.array([overage, underage])
    return costs


def simulate_2_school_costs(
    true_cap1: int,
    true_cap2: int,
    inf_cap1: int,
    inf_cap2: int,
    co: float,
    cu: float,
    iters: int,
    prob: float,
) -> np.ndarray:
    """
    Function that estimates the true total cost corresponding to the system
    of two schools for some decisions on the inflated capacities.
    Returns the cost as an array with the first entry as the
    overage and the second one as the underage cost.

    :param true_cap1: true capacity of the first school
    :param inf_cap1: inflated capacity at the first school
    :param true_cap2: true capacity of the second school
    :param inf_cap2: inflated capacity at the second school
    :param co: per unit cost of overage
    :param cu: per unit cost of underage
    :param iters: number of iterations
    :param prob: probability of dropout
    :return: len 2 array where first entry is overage cost and second is underage over both schools
    """
    sch1_cost = simulate_yield_costs(true_cap1, inf_cap1, co, cu, iters, prob)
    sch2_cost = simulate_yield_costs(true_cap2, inf_cap2, co, cu, iters, prob)
    return sch1_cost + sch2_cost


def optimize_2sch(
    true_cap1: int, true_cap2: int, prob: float, iters: int, co: float, cu: float
):
    """
    Function that performs a grid search optimizing for the best
    pair of inflated capacities in terms of minimum total cost
    to the system.

    :param true_cap1: true capacity of the first school
    :param true_cap2: true capacity of the second school
    :param iters: number of iterations
    :param prob: probability of dropout
    :param co: unit overage cost
    :param cu: unit underage cost
    :return: np.ndarray containing costs from all inflated capacities in [q, q + qp] # TODO: Check that this is correct
    """

    # define intervals of possible amounts by which we can inflate by
    # using qhat / (1-prob) <= q
    int_length1 = int(true_cap1 * prob / (1 - prob))
    int_length2 = int(true_cap2 * prob / (1 - prob))

    # values we try for inflated capacities
    values1 = np.arange(true_cap1, true_cap1 + int_length1 + 1)
    values2 = np.arange(true_cap2, true_cap2 + int_length2 + 1)
    avg_costs = np.zeros(shape=(int_length1, int_length2))

    for qhat1 in values1:
        for qhat2 in values2:
            costs = simulate_2_school_costs(
                true_cap1, true_cap2, qhat1, qhat2, co, cu, iters, prob
            )
            avg_costs[qhat1 - true_cap1, qhat2 - true_cap2] = costs[0] + costs[1]

    return avg_costs


def heuristic_set_capacity(total_cap: int, co: float, cu: float, prob: float):
    """
    Function that computes the total inflation given the heuristic we use
    to construct a long binomial and look for the approximate inverse
    of the CDF for the value equal to the critical ratio.
    Returns an integer, the total inflation.

    :total_cap: total capacity we are computing overage-underage costs with respect to
    :co: per unit cost of overage
    :cu: per unit cost of underage
    :prob: probability of dropout
    """

    int_length = int(total_cap * prob / (1 - prob))
    values = np.arange(total_cap, total_cap + int_length + 1)
    ratio = float(co) / (cu + co)

    best_val = values[0]
    for qhat in values:
        if binom.cdf(total_cap, qhat, 1 - prob) >= ratio:
            best_val = qhat
        else:
            break
    return best_val


def compute_cdf_inflate_sep(
    cap1: int, cap2: int, co: float, cu: float, prob: float
) -> float:
    """
    Function that computes the CDF corresponding to the combined binomial,
    resulting from adding up the inflated capacities, at the sum of true capacities.
    Used to compare whether inflating separately results in more inflation than
    when we are inflating school 2, school of interest, by viewing is as
    integrated in the chain, as opposed to the top of its own chain or separately.
    Returns a float, the value of CDF at the desired point.

    :param cap1: capacity corresponding to the first binomial, so  either the first school or the chain up to our
        school of interest
    :param cap2: capacity
    :param co: per unit cost of overage
    :param cu: per unit cost of underage
    :param prob: probability of dropout
    :return: float with the value of the CDF at the desired point
    """

    # compute the inflated capacities when first school/chain and the second
    # school-school of interest-inflate separately
    qhat1 = heuristic_set_capacity(cap1, co, cu, prob)
    qhat2 = heuristic_set_capacity(cap2, co, cu, prob)

    # compute CDF at the sum of true capacities
    return binom.cdf(cap1 + cap2, qhat1 + qhat2, 1 - prob)


def simulate_multi_school_costs(true_caps, inf_caps, co, cu, iters, prob):
    """
    Function that estimates the true total costs corresponding to each of the
    n schools for some decisions on the inflated capacities. Returns the result
    as an n x 2 matrix, with the overage and underage cost corresponding to
    each school as its rows.

    :true_caps: true capacities of each school
    :inf_caps: inflated capacities at of each school
    :co: per unit cost of overage
    :cu: per unit cost of underage
    :iters: number of iterations
    :prob: probability of dropout

    """

    n = len(true_caps)
    costs = np.zeros((n, 2), dtype=float)

    yld = np.random.binomial(inf_caps, 1-prob, size=(iters, n)).T
    diff = yld - np.tile(true_caps, (iters, 1)).T

    underfill_so_far = np.zeros(iters)
    for k in range(n):
        underfilled_mask = np.where(diff[k, :] < underfill_so_far, 1, 0)
        overfilled_mask = np.where(diff[k, :] > underfill_so_far, 1, 0)
        costs[k, 0] += co*np.sum(overfilled_mask*(diff[k, :] - underfill_so_far))
        costs[k, 1] += cu*np.sum(underfilled_mask*((-1) * diff[k, :] + underfill_so_far))
        remaining_underfill = underfill_so_far - overfilled_mask*diff[k, :] + underfilled_mask * (-1) * diff[k, :]
        underfill_so_far = np.where(remaining_underfill > 0, remaining_underfill, 0)

    costs = costs / iters
    return costs


def heuristic_set_capacity_chain(true_caps, co, cu, p):
    running_true_cap = 0
    running_inf_cap = 0
    inf_caps = []
    for q in true_caps:
        running_true_cap += q
        qhat = heuristic_set_capacity(running_true_cap, co, cu, p) - running_inf_cap
        running_inf_cap += qhat
        inf_caps.append(qhat)
    return np.array(inf_caps)


def heuristic_set_capacity_independent(true_caps, co, cu, p):
    inf_caps = []
    for q in true_caps:
        qhat = heuristic_set_capacity(q, co, cu, p)
        inf_caps.append(qhat)
    return np.array(inf_caps)
