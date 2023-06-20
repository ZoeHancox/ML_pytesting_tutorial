import activations
import numpy as np
from hypothesis import given
from hypothesis.strategies import floats


@given(floats())
def test_sigmoid(x):
    result = activations.sigmoid(x)
    if not np.isnan(x):
      assert 0 <= result <= 1, "Value not between 0 & 1"
