import numpy as np
import pytest
from functools import reduce

from second_quantization import pauli_strings


@pytest.fixture(scope="module", params=["IIII", "XZXY", "Y", "IIYZ"])
def get_string(request):
    """
    Return a pauli string to construct and text against
    """
    return request.param


# +
s_0 = np.eye(2)
s_1 = np.array([[0, 1], [1, 0]])
s_2 = np.array([[0, -1j], [1j, 0]])
s_3 = np.array([[1, 0], [0, -1]])


def build_from_string(get_string):
    options = {"I": s_0, "X": s_1, "Y": s_2, "Z": s_3}
    return reduce(np.kron, [options[letter] for letter in list(get_string)])


# -


def test_string(get_string):
    string = get_string
    matrix = build_from_string(get_string)
    jw_string = list(pauli_strings.PauliDecomposition(matrix).keys())[0]
    assert string == jw_string


def test_output(get_string):
    matrix = build_from_string(get_string)
    jw_string = pauli_strings.PauliDecomposition(matrix)
    assert isinstance(jw_string, dict)
    assert np.all([isinstance(key, str) for key in jw_string.keys()])
    assert np.all(
        [
            isinstance(value, complex) or isinstance(value, float)
            for value in jw_string.values()
        ]
    )
