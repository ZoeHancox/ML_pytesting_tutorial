import activations
import numpy as np
from hypothesis import given
from hypothesis.strategies import floats


@given(floats(min_value=-100, max_value=100))
def test_sigmoid(x):
    result = activations.sigmoid(x)
    if not np.isnan(x):
      assert 0 <= result <= 1, "Value not between 0 & 1"
