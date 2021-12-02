"""Gale Shapley Deferred Acceptance Algorithm implementation."""
from typing import Tuple

import numpy as np


class DeferredAcceptance:
    """Implementation of the student-proposing Gale Shapley Deferred Acceptance Algorithm"""

    def __init__(
        self,
        student_preferences: np.ndarray,
        school_priorities: np.ndarray,
        school_capacities: np.ndarray,
    ):
        """
        Create and set up DA object.

        :param student_preferences: a (num students) x (num schools) array where each student's row corresponds to
            their preference list. The most preferred school is at index 0.
        :param school_priorities: a (num students) x (num schools) array where school_priorities[i,j] contains student
            i's priority at school j. Higher values indicate more highly prioritized.
        :param school_capacities: an array of length (num schools) containing the capacity of each school.
        """
        self.num_students = student_preferences.shape[0]
        self.num_schools = student_preferences.shape[1]
        self.preferences = student_preferences
        self.priorities = school_priorities
        self.capacities = school_capacities
        self.num_ranked = np.full(self.num_students, self.num_schools)

    def da(self) -> dict:
        """
        Execute student proposing deferred acceptance.

        :return: a dictionary where the keys are the schools and the values are the list of students assigned to
            that school.
        """
        current_rank = np.zeros(self.num_students, dtype=int)
        current_students = np.arange(self.num_students, dtype=int)
        tentative_acceptances = {
            sch: np.array([], dtype=int) for sch in range(self.num_schools)
        }

        while len(current_students) > 0:
            (
                current_applications,
                next_rank,
                next_students,
            ) = self._set_up_next_application_round(current_rank, current_students)

            for sch in np.arange(self.num_schools, dtype=int):
                sch_applicants = [
                    current_students[i] for i in np.where(current_applications == sch)
                ]
                if len(sch_applicants) == 0:
                    continue
                sch_applicants = np.append(sch_applicants, tentative_acceptances[sch])

                ordered_priority_index = self._get_priority_ordering(
                    sch, sch_applicants
                )

                accepted, tentative_acceptances = self._update_tentative_acceptances(
                    ordered_priority_index, sch, sch_applicants, tentative_acceptances
                )

                next_students, next_rank = self._update_next_round_students(
                    accepted,
                    next_rank,
                    next_students,
                    ordered_priority_index,
                    sch,
                    sch_applicants,
                )
            current_rank = next_rank
            current_students = next_students

        return tentative_acceptances

    def _set_up_next_application_round(
        self, current_rank: np.ndarray, current_students: np.ndarray
    ) -> Tuple[list, np.ndarray, np.ndarray]:
        """
        Identify the current set of applicants and prepare data structures for subsequent rounds.

        :param current_rank: an array of length (num students) identifying which rank of school each student is
            currently proposing to.
        :param current_students: an array containing the students who are currently unassigned.
        :return: a tuple containing the current applicants for this round, the data structure to keep track of rejected
            students' rank, and the data structure to keep track of rejected students.
        """
        current_applications = [
            self.preferences[i, current_rank[i]] for i in current_students
        ]
        next_rank = np.copy(current_rank)
        next_students = np.copy(current_students)
        return current_applications, next_rank, next_students

    def _get_priority_ordering(self, sch: int, sch_applicants: np.ndarray) -> np.ndarray:
        """
        Order school applicants in order of decreasing priority.

        :param sch: the school number/index
        :param sch_applicants: list of student indexes that are currently applying to sch
        :return: a sorted array of students containing the highest priority student at index 0.
        """
        applicant_priorities = self.priorities[sch_applicants, sch]
        ordered_priority_index = np.argsort(applicant_priorities)[::-1]
        return ordered_priority_index

    def _update_tentative_acceptances(
        self,
        ordered_priority_index: np.ndarray,
        sch: int,
        sch_applicants: np.ndarray,
        tentative_acceptances: dict,
    ) -> Tuple[np.ndarray, dict]:
        """
        Given new proposals, update tentative acceptances to the highest priority students that have applied so far.

        :param ordered_priority_index: a sorted array of students ordered in decreasing priority at sch.
        :param sch: the school currently being considered
        :param sch_applicants: a list of students currently applying to sch.
        :param tentative_acceptances: a dictionary mapping schools to an array of students that they have tentatively
            accepted in previous rounds of proposals
        :return: a list of students accepted to school sch, an updated dictionary of tentative_acceptances
        """
        accepted = sch_applicants[ordered_priority_index[: self.capacities[sch]]]
        tentative_acceptances[sch] = accepted
        return accepted, tentative_acceptances

    def _update_next_round_students(
        self,
        accepted: np.ndarray,
        next_rank: np.ndarray,
        next_students: np.ndarray,
        ordered_priority_index: np.ndarray,
        sch: int,
        sch_applicants: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update status of students who were rejected in the current round of proposals to school sch.

        :param accepted: list of students given tentative acceptances to school sch in this round
        :param next_rank: array containing the rank of schools students should apply to next, if rejected
        :param next_students: array of students who are currently unassigned after this round of proposals
        :param ordered_priority_index: a sorted array of students ordered in decreasing priority at sch.
        :param sch: the school currently being considered
        :param sch_applicants: a list of students currently applying to sch in this round of proposals
        :return: updated array of students who are unassigned after this round of proposals, updated array containing
            the rank of schools students should apply to next, if rejected
        """
        rejected = sch_applicants[ordered_priority_index[self.capacities[sch]:]] # noqa
        next_rank[rejected] += 1
        next_students = next_students[~np.isin(next_students, accepted)]
        next_students = np.append(
            next_students, rejected[~np.isin(rejected, next_students)]
        )
        exhausted_list = np.where(np.greater_equal(next_rank, self.num_ranked))
        if len(exhausted_list) > 0:
            next_students = next_students[~np.isin(next_students, exhausted_list)]
        return next_students, next_rank
