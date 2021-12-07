"""Basic class to generate multinomial logit preferences."""
import numpy as np


class MNL:
    """Class to generate multinomial logit (MNL) preferences where all students have the same utilities."""

    def __init__(self, utilities: np.ndarray, num_students: int = 1):
        """
        Create MNL object and calculate base choice probabilities.

        :param utilities: an array containing the utility for each school.
            The utility for school i would be found at utilities[i].
        """
        self.utilities = utilities
        if utilities.ndim == 1:
            self.num_schools = len(utilities)
        else:
            self.num_schools = utilities.shape[1]
        self.num_students = num_students

    def calculate_choice_probabilities(self) -> np.ndarray:
        """
        Calculate the choice probabilities given the utilities and gumbel noise.

        :return: Array containing choice probabilities for each school. The chance that school i is picked first is
            given by the value at index i in the output array.
        """
        utilities_with_noise = self.utilities + np.random.gumbel(
            size=(self.num_students, self.num_schools)
        )
        return np.exp(utilities_with_noise) / np.sum(np.exp(utilities_with_noise), axis=1)[:, None]

    def sample_preference_ordering(self):
        """
        Sample a rank order preference list given choice probabilities by sampling without replacement.

        :return: a ranking over schools where the number at index 0 is the index of the most popular school.
        """
        probabilities = self.calculate_choice_probabilities()

        def sample(row):
            return np.random.choice(self.num_schools, self.num_schools, replace=False, p=row)

        return np.apply_along_axis(sample, axis=1, arr=probabilities)
