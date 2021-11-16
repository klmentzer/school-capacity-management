import numpy as np
import pytest

from ..src.da import DeferredAcceptance


@pytest.fixture
def student_preferences():
    return np.array(
        [
            [0, 1, 2],
            [2, 0, 1],
            [1, 2, 0]
        ]
    ).T


@pytest.fixture
def school_priorities():
    return np.array(
        [
            [2, 0, 1],
            [1, 2, 0],
            [0, 1, 2]
        ]
    ).T


def test_da_set_up_application_round(student_preferences, school_priorities):
    da = DeferredAcceptance(student_preferences, school_priorities, np.ones(3, dtype=int))
    current_rank = np.zeros(3, dtype=int)
    current_students = np.arange(3, dtype=int)
    current_applications, next_rank, next_students = da._set_up_next_application_round(current_rank, current_students)
    assert np.equal(current_rank, next_rank).all()
    assert np.equal(current_students, next_students).all()
    assert current_applications == [0, 1, 2]


def test_da(student_preferences, school_priorities):
    da = DeferredAcceptance(student_preferences, school_priorities, np.ones(3, dtype=int))
    assignment = da.da()
    assert assignment == {0: np.array([0]), 1: np.array([1]), 2: np.array([2])}


def test_da_school_proposing(student_preferences, school_priorities):
    da = DeferredAcceptance(school_priorities, student_preferences, np.ones(3, dtype=int))
    assignment = da.da()
    assert assignment == {0: np.array([1]), 1: np.array([2]), 2: np.array([0])}


@pytest.fixture
def student_preferences_bigger():
    return np.array(
        [
            [0, 1, 2, 3],
            [0, 3, 2, 1],
            [1, 0, 2, 3],
            [3, 1, 2, 0]
        ]
    )


@pytest.fixture
def school_priorities_bigger():
    return np.array(
        [
            [2, 3, 1, 0],
            [2, 0, 3, 1],
            [1, 2, 3, 0],
            [2, 1, 0, 3]
        ]
    ).T


def test_da_get_priority_ordering(student_preferences_bigger, school_priorities_bigger):
    da = DeferredAcceptance(student_preferences_bigger, school_priorities_bigger, np.ones(4, dtype=int))
    ordered_priority_index = da._get_priority_ordering(2, np.array([1, 2]))
    expected = np.array([0, 1])
    assert np.equal(ordered_priority_index, expected).all()


def test_da_update_tentative_acceptances(student_preferences_bigger, school_priorities_bigger):
    da = DeferredAcceptance(student_preferences_bigger, school_priorities_bigger, np.ones(4, dtype=int))
    accepted, tentative_acceptances = da._update_tentative_acceptances(np.array([0, 1]), 2, np.array([1, 2]), {})
    assert accepted == np.array([1])
    assert tentative_acceptances == {2: np.array(1)}


def test_da_update_next_round_students(student_preferences_bigger, school_priorities_bigger):
    da = DeferredAcceptance(student_preferences_bigger, school_priorities_bigger, np.ones(4, dtype=int))
    accepted = np.array([1])
    next_rank = np.zeros(4, dtype=int)
    next_students = np.array([1, 2])
    ordered_priority_index = np.array([0, 1])
    sch_applicants = np.array([1, 2])
    next_students, next_rank = da._update_next_round_students(accepted, next_rank, next_students,
                                                              ordered_priority_index, 2,
                                                              sch_applicants)
    assert np.equal(next_students, np.array([2])).all()
    assert np.equal(next_rank, np.array([0, 0, 1, 0])).all()


def test_da_larger(student_preferences_bigger, school_priorities_bigger):
    da = DeferredAcceptance(student_preferences_bigger, school_priorities_bigger, np.ones(4, dtype=int))
    assignment = da.da()
    assert assignment == {0: np.array([2]), 1: np.array([3]), 2: np.array([0]), 3: np.array([1])}


def test_da_larger_many_to_one(student_preferences_bigger, school_priorities_bigger):
    da = DeferredAcceptance(student_preferences_bigger, school_priorities_bigger, 2 * np.ones(4, dtype=int))
    assignment = da.da()
    assert len(assignment) == 4
    assert np.equal(assignment[0], np.array([0, 1])).all()
    assert np.equal(assignment[1], np.array([2])).all()
    assert np.equal(assignment[2], np.array([])).all()
    assert np.equal(assignment[3], np.array([3])).all()
