import numpy as np
import pytest

from ..src.scoring_rules import CopelandScoreSchools, BordaScoreSchools


@pytest.fixture
def schools():
    return [101, 202, 303]


@pytest.fixture
def prefs():
    return [[101], [202, 101], [101, 303, 202]]


def test_copeland_unranked_schools(schools, prefs):
    cs = CopelandScoreSchools(schools, prefs)
    cs._add_unranked_schools([0, 2])
    expected_scores = np.array([[0, 1, 0], [0, 0, 0], [0, 1, 0]])
    assert np.equal(cs.scores, expected_scores).all()


def test_copeland_ranked_schools(schools, prefs):
    cs = CopelandScoreSchools(schools, prefs)
    cs._add_ranked_schools([0, 2])
    expected_scores = np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]])
    assert np.equal(cs.scores, expected_scores).all()


def test_copeland_score_preferences(schools, prefs):
    cs = CopelandScoreSchools(schools, prefs)
    cs._score_preferences()
    expected_scores = np.array([[0, 2, 3], [1, 0, 1], [0, 1, 0]])
    assert np.equal(cs.scores, expected_scores).all()


def test_copeland_determine_ordering(schools, prefs):
    cs = CopelandScoreSchools(schools, prefs)
    cs.scores = np.array([[0, 2, 3], [1, 0, 1], [0, 1, 0]])
    ordering = cs._determine_ordering()
    expected_ordering = [(101, 2), (303, 0), (202, 0)]
    assert ordering == expected_ordering


def test_copeland_score_ordering(schools, prefs):
    cs = CopelandScoreSchools(schools, prefs)
    ordering = cs.copeland_score_ordering()
    expected_ordering = [(101, 2), (303, 0), (202, 0)]
    assert ordering == expected_ordering


def test_borda_remove_duplicates_retaining_order(schools, prefs):
    bs = BordaScoreSchools(schools, prefs)
    actual = bs._remove_duplicates_retaining_order([2, 1, 1, 3, 2, 5, 3, 4])
    expected = [2, 1, 3, 5, 4]
    assert actual == expected


def test_calc_borda_score(schools, prefs):
    bs = BordaScoreSchools(schools, prefs)
    score = bs._calc_borda_score()
    expected_score = np.array([5, 2, 1])
    assert np.equal(score, expected_score).all()


def test_borda_score_ordering(schools, prefs):
    bs = BordaScoreSchools(schools, prefs)
    order = bs.borda_score_ordering()
    expected_order = [(101, 5), (202, 2), (303, 1)]
    assert order == expected_order
