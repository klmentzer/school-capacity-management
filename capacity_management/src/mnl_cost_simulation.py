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
        super().__init__(co, cu, iters, prob, true_caps, inf_strategy)
        self.mus = mus
        self.num_schools = len(mus)
        self.num_students = int(sum(true_caps)/(1-prob))
        self.mnl = MNL(mus, self.num_students)  # TODO: determine if this is the correct number of students

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
        assert np.equal(order, np.argsort(r1_cutoffs)).all()
        assert np.greater(r2_cutoffs[:-1], r1_cutoffs[1:]).all()
