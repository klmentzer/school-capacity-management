import numpy as np
from scipy.stats import binom


class CostSimulator:
    def __init__(self, co: float, cu: float, iters: int, prob: float, true_caps: np.ndarray = None, inf_strategy: str = None):
        self.true_caps = true_caps
        self.co = co
        self.cu = cu
        self.iters = iters
        self.prob = prob
        self.allowed_strategies = ['chain', 'independent']
        if inf_strategy is not None and inf_strategy not in self.allowed_strategies:
            raise ValueError(f"Inflation strategy not recognized, please use one of {self.allowed_strategies}")
        self.inf_strategy = inf_strategy

    def simulate_yield_costs(
            self, true_cap: int, inf_cap: int
    ) -> np.ndarray:
        """
        Function that estimates the true cost corresponding to the
        first school for a given true capacity and inflated capacity.
        Returns the cost as an array with the first entry as the
        overage and the second one as the underage cost.

        :param true_cap: true capacity of the school
        :param inf_cap: inflated capacity
        :return: len 2 array where first entry is overage cost and second is underage for the one school
        """
        yld = np.full(self.iters, inf_cap, dtype=float) - np.random.binomial(
            inf_cap, self.prob, size=self.iters
        )
        diff = yld - np.full(self.iters, true_cap, dtype=float)
        overage = self.co * sum(np.where(diff > 0, diff, 0)) / self.iters
        underage = self.cu * sum(np.where(-1 * diff > 0, -1 * diff, 0)) / self.iters
        costs = np.array([overage, underage])
        return costs

    def simulate_2_school_costs(
            self,
            true_cap1: int,
            true_cap2: int,
            inf_cap1: int,
            inf_cap2: int,
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
        :return: len 2 array where first entry is overage cost and second is underage over both schools
        """
        sch1_cost = self.simulate_yield_costs(true_cap1, inf_cap1)
        sch2_cost = self.simulate_yield_costs(true_cap2, inf_cap2)
        return sch1_cost + sch2_cost

    def optimize_2sch(
            self, true_cap1: int, true_cap2: int,
    ):
        """
        Function that performs a grid search optimizing for the best
        pair of inflated capacities in terms of minimum total cost
        to the system.

        :param true_cap1: true capacity of the first school
        :param true_cap2: true capacity of the second school
        :return: np.ndarray containing costs from all inflated capacities in [q, q + qp/(1-p)]
        """

        # define intervals of possible amounts by which we can inflate by
        # using qhat / (1-prob) <= q
        int_length1 = int(true_cap1 * self.prob / (1 - self.prob))
        int_length2 = int(true_cap2 * self.prob / (1 - self.prob))

        # values we try for inflated capacities
        values1 = np.arange(true_cap1, true_cap1 + int_length1 + 1)
        values2 = np.arange(true_cap2, true_cap2 + int_length2 + 1)
        avg_costs = np.zeros(shape=(int_length1+1, int_length2+1))

        for qhat1 in values1:
            for qhat2 in values2:
                costs = self.simulate_2_school_costs(
                    true_cap1, true_cap2, qhat1, qhat2
                )
                avg_costs[qhat1 - true_cap1, qhat2 - true_cap2] = costs[0] + costs[1]

        return avg_costs

    def heuristic_set_capacity(self, total_cap: int):
        """
        Function that computes the total inflation given the heuristic we use
        to construct a long binomial and look for the approximate inverse
        of the CDF for the value equal to the critical ratio.
        Returns an integer, the total inflation.

        :param total_cap: total capacity we are computing overage-underage costs with respect to
        """

        int_length = int(total_cap * self.prob / (1 - self.prob))
        values = np.arange(total_cap, total_cap + int_length + 1)
        ratio = float(self.co) / (self.cu + self.co)

        best_val = values[0]
        for qhat in values:
            if binom.cdf(total_cap, qhat, 1 - self.prob) >= ratio:
                best_val = qhat
            else:
                break
        return best_val

    def compute_cdf_inflate_sep(
            self, cap1: int, cap2: int,
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
        :return: float with the value of the CDF at the desired point
        """

        # compute the inflated capacities when first school/chain and the second
        # school-school of interest-inflate separately
        qhat1 = self.heuristic_set_capacity(cap1)
        qhat2 = self.heuristic_set_capacity(cap2)

        # compute CDF at the sum of true capacities
        return binom.cdf(cap1 + cap2, qhat1 + qhat2, 1 - self.prob)

    def simulate_multi_school_costs(self, inf_caps):
        """
        Function that estimates the true total costs corresponding to each of the
        n schools for some decisions on the inflated capacities. Returns the result
        as an n x 2 matrix, with the overage and underage cost corresponding to
        each school as its rows.

        :param inf_caps: inflated capacities at of each school
        """

        n = len(self.true_caps)
        costs = np.zeros((n, 2), dtype=float)

        yld = np.random.binomial(inf_caps, 1 - self.prob, size=(self.iters, n)).T
        diff = yld - np.tile(self.true_caps, (self.iters, 1)).T

        underfill_so_far = np.zeros(self.iters)
        for k in range(n):
            underfilled_mask = np.where(diff[k, :] < underfill_so_far, 1, 0)
            overfilled_mask = np.where(diff[k, :] > underfill_so_far, 1, 0)
            costs[k, 0] += self.co * np.sum(overfilled_mask * (diff[k, :] - underfill_so_far))
            costs[k, 1] += self.cu * np.sum(underfilled_mask * ((-1) * diff[k, :] + underfill_so_far))
            remaining_underfill = underfill_so_far - overfilled_mask * diff[k, :] + underfilled_mask * (-1) * diff[k, :]
            underfill_so_far = np.where(remaining_underfill > 0, remaining_underfill, 0)

        self.costs = costs / self.iters
        self.underfill = underfill_so_far

    def heuristic_set_capacity_chain(self):
        running_true_cap = 0
        running_inf_cap = 0
        inf_caps = []
        for q in self.true_caps:
            running_true_cap += q
            qhat = self.heuristic_set_capacity(running_true_cap) - running_inf_cap
            running_inf_cap += qhat
            inf_caps.append(qhat)
        return np.array(inf_caps)

    def heuristic_set_capacity_independent(self):
        inf_caps = []
        for q in self.true_caps:
            qhat = self.heuristic_set_capacity(q)
            inf_caps.append(qhat)
        return np.array(inf_caps)

    def evaluation_metrics(self, inf_caps):
        metrics = {}
        metrics["total_cost"] = np.sum(self.costs)
        metrics["school_costs"] = np.sum(self.costs, axis=1)
        metrics["school_overage_costs"] = self.costs[:, 0]
        metrics["school_underage_costs"] = self.costs[:, 1]
        metrics["avg_underfill/movers"] = np.mean(self.underfill)
        metrics["raw_inflation"] = inf_caps - self.true_caps
        metrics["pct_inflation"] = metrics["raw_inflation"] / self.true_caps
        return metrics

    def simulate(self):
        capacity_func = getattr(self, f"heuristic_set_capacity_{self.inf_strategy}")
        inf_caps = capacity_func()
        self.simulate_multi_school_costs(inf_caps)
        return self.evaluation_metrics(inf_caps)


