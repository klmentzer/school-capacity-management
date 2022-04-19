import numpy as np
from .cost_simulation import CostSimulator
from .mnl import MNL
from .two_round_simulation import run_2_round_assignment


class MNLCostSimulator(CostSimulator):
    def __init__(
        self,
        co: float,
        cu: float,
        iters: int,
        prob: float,
        true_caps: np.ndarray,
        inf_strategy: str,
        mus: np.ndarray,
    ):
        super().__init__(co, cu, iters, prob, true_caps)
        self.inf_strategy = inf_strategy
        self.mus = mus  # assume sorted in non-increasing order
        self.num_schools = len(mus)
        self.num_students = int(sum(true_caps)/(1-prob))
        self.mnl = MNL(mus, self.num_students)  # TODO: determine if this is the correct number of students
        self.mnl_coeff = self.calc_mnl_coefficients()
        if self.inf_strategy == 'mnl':
            self.validate_order_condition()

    def calc_mnl_coefficients(self) -> np.ndarray:
        """
        Calculate the MNL coefficients for each pair of schools in the optimality condition.

        :return: np.ndarray where coeff[i, k] is c_i^k, the fraction of seats at i that will be taken by students at
            school k
        """
        n = len(self.true_caps)
        exp_mu = np.exp(self.mus)
        coeffs = np.ones((n, n))
        for i in range(n):
            coeffs[i, i] = 1
            for k in range(i + 1, n):
                coeffs[i, k] = np.sum(
                    [exp_mu[k] / np.sum(exp_mu[j:]) * coeffs[i, j] for j in range(k)]
                )
            coeffs[i, :i] = 0
        return coeffs

    def validate_order_condition(self):
        capacity_func = getattr(self, f"heuristic_set_capacity_{self.inf_strategy}")
        inf_caps = capacity_func()
        preferences = self.mnl.sample_preference_ordering()
        priorities = np.random.uniform(size=(self.num_students, self.num_schools))
        r1, r2 = run_2_round_assignment(preferences, priorities, inf_caps, self.prob, return_r1_assignment=True)
        r1_cutoffs = np.zeros(self.num_schools)
        r2_cutoffs = np.zeros(self.num_schools)
        for sch, students in r1.items():
            r1_cutoffs[sch] = np.min(priorities[students, sch])
            r2_cutoffs[sch] = np.min(priorities[r2[sch], sch])
        order = np.argsort(self.mus)
        if not np.equal(order, np.argsort(r1_cutoffs)).all() or not np.greater(r2_cutoffs[:-1], r1_cutoffs[1:]).all():
            raise ValueError("The provided mus do not satisfy the order condition.")

    def heuristic_set_capacity_mnl(self):
        inf_caps = np.zeros(len(self.true_caps), dtype=int)
        for i, q in enumerate(self.true_caps):
            weighted_true_caps = sum(np.multiply(self.mnl_coeff[:i+1, i], self.true_caps[:i+1]))
            weighted_inf_caps = np.sum(self.mnl_coeff[:i, i]*inf_caps[:i])
            qhat = self.heuristic_set_capacity(weighted_true_caps) - weighted_inf_caps
            inf_caps[i] = int(qhat)
        return np.array(inf_caps)

    def simulate(self, inf_strategy=None):
        inf_strategy = inf_strategy or self.inf_strategy
        capacity_func = getattr(self, f"heuristic_set_capacity_{inf_strategy}")
        inf_caps = capacity_func()

        self.underfill = np.zeros(self.num_schools)
        self.overfill = np.zeros(self.num_schools)
        self.num_movers = np.zeros(self.num_schools)
        self.num_dropout = None  # np.zeros(self.num_schools)
        for t in range(self.iters):
            preferences = self.mnl.sample_preference_ordering()
            priorities = np.tile(np.random.uniform(size=self.num_students), (self.num_schools, 1)).T
            r1, r2 = run_2_round_assignment(preferences, priorities, inf_caps, self.prob, r2_capacities=self.true_caps, return_r1_assignment=True)

            num_assigned = np.zeros(self.num_schools)
            for k, v in r2.items():
                num_assigned[k] = len(v)

            self.overfill += np.where(num_assigned > self.true_caps, num_assigned-self.true_caps, 0)/ self.iters
            self.underfill += np.where(num_assigned < self.true_caps, self.true_caps - num_assigned, 0)/ self.iters

            # number of students who move up to each school (assigned in round 2 but not in round 1)
            for k, v in r2.items():
                self.num_movers[k] += len(set(v)-set(r1[k]))/self.iters
                # self.num_dropout[k] += len(set(r1[k])-set(v))/self.iters

        self.costs = np.vstack([self.co*self.overfill, self.cu*self.underfill]).T
        return self.evaluation_metrics(inf_caps)

    def evaluation_metrics(self, inf_caps):
        metrics = super().evaluation_metrics(inf_caps)
        metrics['movers'] = self.num_movers
        metrics['dropout'] = self.num_dropout
        return metrics

    def chaining_inflation_difference(self) -> float:
        """
        Calculate the percentage increase in inflation percentage (not total true capacity)

        :return: float representing the (chained % inflation)/(independent % inflation)
        """
        independent = self.simulate(inf_strategy='independent')
        chained = self.simulate(inf_strategy='mnl')
        return chained['pct_inflation'] / independent['pct_inflation']

