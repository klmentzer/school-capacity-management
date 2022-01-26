from ..src.da_deterministic_dropout import DeterministicDropoutDA
from .test_da import student_preferences, school_priorities
import numpy as np


def test_sum_binary_search(student_preferences, school_priorities):
    ddda = DeterministicDropoutDA(
        student_preferences,
        school_priorities,
        np.full(3, 3, dtype=int),
        np.zeros((3, 3), dtype=int),
    )
    cutoff = ddda._sum_binary_search([1, 1, 1, 1], 0)
    assert cutoff == 3
    priority_ordered_stayers = [0, 1, 1, 0, 1, 0, 1, 1]
    cutoff = ddda._sum_binary_search(priority_ordered_stayers, 0)
    assert cutoff == 6


def test_sum_binary_search_fractional(student_preferences, school_priorities):
    ddda = DeterministicDropoutDA(
        student_preferences,
        school_priorities,
        np.full(3, 1, dtype=int),
        np.zeros((3, 3), dtype=int),
    )
    cutoff = ddda._sum_binary_search([0.5, 0, 0.3, 0.5, 0.5, 0.5], 0)
    assert cutoff == 3


def test_zero_dropout_da(student_preferences, school_priorities):
    ddda = DeterministicDropoutDA(
        student_preferences,
        school_priorities,
        np.ones(3, dtype=int),
        np.zeros((3, 3), dtype=int),
    )
    assignment = ddda.da()
    assert assignment == {0: np.array([0]), 1: np.array([1]), 2: np.array([2])}


def test_deterministic_dropout_da():
    preferences = np.array([[0, 1], [0, 1], [0, 1]])
    priorities = np.array([[2, 1, 0], [2, 1, 0]]).T
    dropout = np.array([[1, 0, 0], [1, 0, 0]]).T
    ddda = DeterministicDropoutDA(
        preferences, priorities, np.ones(2, dtype=int), dropout
    )
    assignment = ddda.da()
    assert np.equal(assignment[0], np.array([0, 1])).all()
    assert np.equal(assignment[1], np.array([2])).all()
