
import pytest
import numpy as np
from calc_mean import calculate_mean

def test_calculate_mean():
    numbers = [1, 2, 3, 4, 5]
    assert calculate_mean(numbers) == 3.0

def test_calculate_mean_empty_list():
    numbers = []
    assert calculate_mean(numbers) == None

def test_calculate_mean_single_number():
    numbers = [10]
    assert calculate_mean(numbers) == 10.0

def test_calculate_mean_negative_numbers():
    numbers = [-1, -2, -3, -4, -5]
    # you can also use np.allclose() to assert whether the answers are close
    assert np.allclose(calculate_mean(numbers), -3.0)
