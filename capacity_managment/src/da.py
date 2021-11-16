"""Gale Shapley Deferred Acceptance Algorithm implementation."""
import numpy as np


class DeferredAcceptance:
    """Implementation of the student-proposing Gale Shapley Deferred Acceptance Algorithm"""

    def __init__(self, student_preferences, school_priorities, school_capacities):
        self.num_students = student_preferences.shape[0]
        self.num_schools = student_preferences.shape[1]
        self.preferences = student_preferences
        self.priorities = school_priorities
        self.capacities = school_capacities
        self.num_ranked = np.full(self.num_students, self.num_schools)

    def da(self):
        current_rank = np.zeros(self.num_students, dtype=int)
        current_students = np.arange(self.num_students, dtype=int)
        tentative_acceptances = {sch: np.array([], dtype=int) for sch in range(self.num_schools)}

        while len(current_students) > 0:
            current_applications, next_rank, next_students = self._set_up_next_application_round(current_rank,
                                                                                                 current_students)

            for sch in np.arange(self.num_schools, dtype=int):
                sch_applicants = [current_students[i] for i in np.where(current_applications == sch)]
                if len(sch_applicants) == 0:
                    continue
                sch_applicants = np.append(sch_applicants, tentative_acceptances[sch])

                ordered_priority_index = self._get_priority_ordering(sch, sch_applicants)

                accepted, tentative_acceptances = self._update_tentative_acceptances(ordered_priority_index, sch,
                                                                                     sch_applicants,
                                                                                     tentative_acceptances)

                next_students, next_rank = self._update_next_round_students(accepted, next_rank, next_students,
                                                                            ordered_priority_index, sch, sch_applicants)
            current_rank = next_rank
            current_students = next_students

        return tentative_acceptances

    def _set_up_next_application_round(self, current_rank, current_students):
        current_applications = [
            self.preferences[i, current_rank[i]]
            for i in current_students
        ]
        next_rank = np.copy(current_rank)
        next_students = np.copy(current_students)
        return current_applications, next_rank, next_students

    def _get_priority_ordering(self, sch, sch_applicants):
        applicant_priorities = self.priorities[sch_applicants, sch]
        ordered_priority_index = np.argsort(applicant_priorities)
        return ordered_priority_index

    def _update_tentative_acceptances(self, ordered_priority_index, sch, sch_applicants,
                                      tentative_acceptances):
        accepted = sch_applicants[ordered_priority_index[:self.capacities[sch]]]
        tentative_acceptances[sch] = accepted
        return accepted, tentative_acceptances

    def _update_next_round_students(self, accepted, next_rank, next_students, ordered_priority_index, sch,
                                    sch_applicants):
        rejected = sch_applicants[ordered_priority_index[self.capacities[sch]:]]
        next_rank[rejected] += 1
        next_students = next_students[~np.isin(next_students, accepted)]
        next_students = np.append(next_students, rejected[~np.isin(rejected, next_students)])
        exhausted_list = np.where(np.greater_equal(next_rank, self.num_ranked))
        if len(exhausted_list) > 0:
            next_students = next_students[~np.isin(next_students, exhausted_list)]
        return next_students, next_rank
