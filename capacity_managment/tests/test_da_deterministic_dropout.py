from ..src.da_deterministic_dropout import DeterministicDropoutDA
from test_da import student_preferences, school_priorities
import numpy as np

def test_sum_binary_search():
    ddda = DeterministicDropoutDA(student_preferences, school_priorities, np.ones(3, dtype=int))
