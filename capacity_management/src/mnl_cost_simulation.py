import numpy as np
from .cost_simulation import CostSimulator


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
