from .da import DeferredAcceptance
import numpy as np


class DeterministicDropoutDA(DeferredAcceptance):
    """ DA """

    def __init__(self, student_preferences, school_priorities, school_capacities, dropout_matrix):
        super().__init__(student_preferences, school_priorities, school_capacities)
        self.stayer_matrix = 1- dropout_matrix

    def _sum_binary_search(self, priority_ordered_stayers, sch):
        low = 0
        high = len(priority_ordered_stayers)
        while low < high:
            mid = (high + low) // 2
            if np.sum(priority_ordered_stayers[:mid + 1]) < self.capacities[sch]:
                low = mid + 1
            elif np.sum(priority_ordered_stayers[:mid + 1]) == self.capacities[sch] and high - low == 1:
                if np.sum(priority_ordered_stayers[:high + 1]) == self.capacities[sch]:
                    return high +1
                else:
                    return low +1
            elif np.sum(priority_ordered_stayers[:mid + 1]) == self.capacities[sch]:
                low = mid
            else:
                high = mid - 1
        return high + 1

    def _update_tentative_acceptances(self, ordered_priority_index, sch, sch_applicants,
                                      tentative_acceptances):
        priority_ordered_stayers = self.stayer_matrix[ordered_priority_index, sch]
        self.cutoff_index = self._sum_binary_search(priority_ordered_stayers, sch)
        accepted = sch_applicants[ordered_priority_index[:self.cutoff_index]]
        tentative_acceptances[sch] = accepted
        return accepted, tentative_acceptances

    def _update_next_round_students(self, accepted, next_rank, next_students, ordered_priority_index, sch,
                                    sch_applicants):
        rejected = sch_applicants[ordered_priority_index[self.cutoff_index:]]
        next_rank[rejected] += 1
        next_students = next_students[~np.isin(next_students, accepted)]
        next_students = np.append(next_students, rejected[~np.isin(rejected, next_students)])
        exhausted_list = np.where(np.greater_equal(next_rank, self.num_ranked))
        if len(exhausted_list) > 0:
            next_students = next_students[~np.isin(next_students, exhausted_list)]
        return next_students, next_rank

