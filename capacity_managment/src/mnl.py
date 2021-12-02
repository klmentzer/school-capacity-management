"""Basic class to generate multinomial logit preferences."""
import numpy as np


class MNL:
    """Class to generate multinomial logit (MNL) preferences where all students have the same utilities."""

    def __init__(self, utilities: np.ndarray):
        """
        Create MNL object and calculate base choice probabilities.

        :param utilities: an array containing the utility for each school.
            The utility for school i would be found at utilities[i].
        """
        self.utilities = utilities

    def calculate_choice_probabilities(self) -> np.ndarray:
        """
        Calculate the choice probabilities given the utilities and gumbel noise.

        :return: Array containing choice probabilities for each school. The chance that school i is picked first is
            given by the value at index i in the output array.
        """
        utilities_with_noise = self.utilities + np.random.gumbel(
            size=len(self.utilities)
        )
        return np.exp(utilities_with_noise) / sum(np.exp(utilities_with_noise))

    def sample_preference_ordering(self):
        """
        Sample a rank order preference list given choice probabilities by sampling without replacement.

        :return: a ranking over schools where the number at index 0 is the index of the most popular school.
        """
        probabilities = self.calculate_choice_probabilities()
        return np.random.choice(
            len(self.utilities), len(self.utilities), replace=False, p=probabilities
        )
