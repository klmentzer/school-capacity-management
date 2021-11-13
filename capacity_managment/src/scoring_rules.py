"""Scoring rules to determine school popularity given student preferences."""
from typing import Union, List, Tuple

import numpy as np
import pandas as pd


class CopelandScoreSchools:
    """Score and order schools using the Copeland Rule."""

    def __init__(self, schools: Union[list, pd.Series], prefs: List[List]):
        """
        Create Copeland scoring object.

        :param schools: list of all available schools, including all ranked in preferences
        :param prefs: List of lists, where each inner list is a student's preference ranking over schools
        """
        self.schools = schools
        self.num_schools = len(schools)
        self.school2idx = dict(zip(schools, range(self.num_schools)))
        self.idx2school = dict(zip(range(self.num_schools), schools))
        self.prefs = prefs
        self.scores = np.zeros((self.num_schools, self.num_schools))

    def _add_unranked_schools(self, ranked_idxs: List[int]):
        """
        For a given preference ranking, give all ranked schools a victory over all unranked schools.

        :param ranked_idxs: the  school indices for the current ranking, from most preferred to least preferred
        """
        unranked = [i for i in range(self.num_schools) if i not in ranked_idxs]
        self.scores[np.ix_(ranked_idxs, unranked)] += 1

    def _add_ranked_schools(self, ranked_idxs: List[int]):
        """
        For a given preference ranking, give ranked schools a victory over schools ranked lower.

        :param ranked_idxs: the  school indices for the current ranking, from most preferred to least preferred
        """
        for list_idx, school_idx in enumerate(ranked_idxs):
            lower_ranked = ranked_idxs[list_idx + 1 :] # noqa
            self.scores[school_idx, lower_ranked] += 1

    def _score_preferences(self):
        """
        Calculate the scores for all students' rankings in the preferences.

        Score using (num_schools) x (num_schools) np.ndarray where scores[i, j] = the number of times i has been
        ranked before j in any student's preference list .
        """
        for ranking in self.prefs:
            ranked_idxs = [self.school2idx[j] for j in ranking]
            self._add_ranked_schools(ranked_idxs)
            self._add_unranked_schools(ranked_idxs)

    def _determine_ordering(self) -> List[Tuple]:
        """
        Given the scores, calculate the ordering of schools according to the Copeland rule.

        :return: A list of schools from most popular to least popular according to the Copeland scoring rule. The first
            number in the tuple is the school identifier, and the second is the number of pairwise victories it has over
            other schools.
        """
        winners = np.greater(self.scores, self.scores.T)
        num_wins = np.sum(winners, axis=1)
        idx_ordering = np.argsort(num_wins)[::-1]
        school_ordering = [(self.idx2school[i], num_wins[i]) for i in idx_ordering]
        return school_ordering

    def copeland_score_ordering(self) -> List[Tuple]:
        """
        Use the Copeland rule to determine the most popular schools in a given preference profile.

        :return: A list of schools from most popular to least popular according to the Copeland scoring rule. The first
            number in the tuple is the school identifier, and the second is the number of pairwise victories it has over
            other schools.
        """
        self._score_preferences()
        return self._determine_ordering()


class BordaScoreSchools:
    """Score and order schools using the Borda Rule."""

    def __init__(self, schools: Union[list, pd.Series], prefs):
        """
        Create Borda scoring object.

        :param schools: list of all available schools, including all ranked in preferences
        :param prefs: List of lists, where each inner list is a student's preference ranking over schools
        """
        self.num_schools = len(schools)
        self.school2idx = dict(zip(schools, range(self.num_schools)))
        self.idx2school = dict(zip(range(self.num_schools), schools))
        self.prefs = prefs

    @staticmethod
    def _remove_duplicates_retaining_order(ranking: List) -> List:
        """
        Modify preferences to take the first time a school appears in a preference list.

        Due to the presence of multiple programs in some schools, some schools may appear twice in a given preference
        ranking. A future modification might be to rank programs instead of schools, or to only use general education
        programs in the computation

        :param ranking: Preference list over schools, from most preferred to least preferred
        :return: A preference list maintaining order but only preserving the first appearance of each school in the
            preference ordering.
        """
        seen = set()
        seen_add = seen.add
        return [x for x in ranking if not (x in seen or seen_add(x))]

    def _calc_borda_score(self) -> np.ndarray:
        """
        Calculate the Borda school of each school given the preference profile.

        :return: An array containing the number of points accumulated by each school according to Borda and the given
            preference profile.
        """
        score = np.zeros(self.num_schools)
        points = range(self.num_schools)[::-1]
        for ranking in self.prefs:
            first_occurrence = self._remove_duplicates_retaining_order(ranking)
            for rank, school in enumerate(first_occurrence):
                score[self.school2idx[school]] += points[rank]
        return score

    def borda_score_ordering(self) -> List[Tuple]:
        """
        Use the Borda rule to determine the most popular schools in a given preference profile.

        :return: A list of schools from most popular to least popular according to the Borda scoring rule. The first
            number in the tuple is the school identifier, and the second is the number of points the school received.
        """
        score = self._calc_borda_score()
        borda_idx_order = np.argsort(score)[::-1]
        borda_school_order = [(self.idx2school[i], score[i]) for i in borda_idx_order]
        return borda_school_order
