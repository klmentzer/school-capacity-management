from typing import Tuple, Union

import numpy as np

from capacity_management.src.da import DeferredAcceptance


def simulate_dropout(assignment: dict, p: float) -> Tuple[dict, list]:
    """
    Simulate students dropping out independently with probability p

    :param assignment: dictionary mapping schools to a list of students assigned to that school
    :param p: probability of a student dropping out
    :return: the new assignment after dropout, the list of students who drop out
    """
    after_dropout = {}
    leaving_list = []
    for school, students in assignment.items():
        leaving = np.array(np.random.binomial(size=len(students), n=1, p=p), dtype=bool)
        after_dropout[school] = students[~leaving]
        leaving_list = np.append(leaving_list, students[leaving])
    return after_dropout, [int(i) for i in leaving_list]


def generate_rnd2_preferences(
    preferences: np.ndarray, leaving: list, num_schools: int, num_students: int
) -> np.ndarray:
    """
    Create round 2 preferences taking into account dropout.

    We create a dummy school that represents all unassigned or leaving students. Students who depart put this school
    as their first choice, and all other students have it as their last choice.

    :param preferences: a (num students) x (num schools) array where each student's row corresponds to
            their preference list. The most preferred school is at index 0.
    :param leaving: a list of students who drop out
    :param num_schools: number of schools
    :param num_students: number of students
    :return: preferences updated to assign dropout students to dummy school.
    """
    rnd2_preferences = np.append(
        preferences, np.full((num_students, 1), num_schools, dtype=int), 1
    )
    rnd2_preferences[leaving, -1] = rnd2_preferences[leaving, 0]
    rnd2_preferences[leaving, 0] = num_schools
    return rnd2_preferences


def generate_rnd2_priorities(
    priorities: np.ndarray, after_dropout: dict, num_students: int
) -> np.ndarray:
    """
    Create priorities for second round, taking first round assignment into account.

    For students who do not drop out, give them a priority boost to the school that they were assigned to in the first
    round, so students cannot do worse. Also add an extra priority column for the dummy school.

    :param priorities: a (num students) x (num schools) array where school_priorities[i,j] contains student
        i's priority at school j. Higher values indicate more highly prioritized.
    :param after_dropout: the assignment of students to schools, represented as a dictionary mapping each school to the
        list of students assigned to that school, after simulating dropout
    :param num_students: number of students
    :return: priorities updated to ensure students can only improve in second round
    """
    rnd2_priorities = np.copy(priorities)
    for sch, students in after_dropout.items():
        rnd2_priorities[students, sch] += 1000
    return np.append(rnd2_priorities, np.zeros((num_students, 1), dtype=int), axis=1)


def run_2_round_assignment(
    preferences: np.ndarray, priorities: np.ndarray, r1_capacities: np.ndarray, p: float, r2_capacities: np.ndarray = None, return_r1_assignment: bool = False
) -> Union[dict, Tuple[dict, dict]]:
    """
    Run 2 rounds of assignment, with independent dropout between rounds.

    :param preferences: a (num students) x (num schools) array where each student's row corresponds to
        their preference list. The most preferred school is at index 0.
    :param priorities: a (num students) x (num schools) array where school_priorities[i,j] contains student
        i's priority at school j. Higher values indicate more highly prioritized.
    :param r1_capacities: an array of length (num schools) containing the capacity of each school.
    :param p: probability of a student dropping out
    :return: a dictionary mapping a school to a list of students assigned after round 2
    """
    if r2_capacities is None:
        r2_capacities = r1_capacities
    num_students, num_schools = preferences.shape

    da = DeferredAcceptance(preferences, priorities, r1_capacities)
    rnd1_assignment = da.da()

    after_dropout, leaving = simulate_dropout(rnd1_assignment, p)

    rnd2_preferences = generate_rnd2_preferences(
        preferences, leaving, num_schools, num_students
    )
    rnd2_priorities = generate_rnd2_priorities(priorities, after_dropout, num_students)
    num_assigned_after_dropout = np.zeros(num_schools)
    for k, v in after_dropout.items():
        num_assigned_after_dropout[k] = len(v)
    r2_functional_capacities = np.max(num_assigned_after_dropout, r2_capacities)
    da2 = DeferredAcceptance(
        rnd2_preferences,
        rnd2_priorities,
        np.append(r2_functional_capacities, np.array([num_students])),
    )
    rnd2_assignment = da2.da()
    del rnd2_assignment[num_schools]

    if return_r1_assignment:
        return rnd1_assignment, rnd2_assignment
    return rnd2_assignment
