# Copied from https://github.com/HANTLUK/PauliDecomposition and modified
# to support sympy matrices.

# MIT License
#
# Copyright (c) 2023 Lukas Hantzko
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from collections import defaultdict

import numpy as np
from scipy import sparse
import sympy


def PauliDecomposition(matrix, PauliStringInit=""):
    """
    Computes the Pauli decomposition of a square matrix.

    Iteratively splits tensor factors off and decomposes those smaller
    matrices. This is done using submatrices of the original matrix.
    The Pauli strings are generated in each step.

    Args:
            matrix: Matrix to be decomposed
                            (Preferably numpy array/scipysparse).
            sparse: Whether matrix is in sparse format.
            PauliStringInit: For recursive computation.

    Returns:
            decomposition/outString: String of 1XYZ with their factors.
    """
    # Dimension check
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix is not square.")
    if (qBitDim := np.log(matrix.shape[0]) / np.log(2)) % 1 != 0:
        raise ValueError("Matrix dimension is not a power of 2.")
    I = sympy.I if isinstance(matrix, sympy.Matrix) else 1j  # noqa: E741

    decomposition = defaultdict(lambda: 0)

    # Output for dimension 1
    if qBitDim == 0:
        if matrix[0, 0] != 0.0:
            decomposition[PauliStringInit] = matrix[0, 0]
    else:
        # Calculates the tensor product coefficients via the sliced submatrices.
        # If one of these components is zero that coefficient is ignored.
        halfDim = int(2 ** (qBitDim - 1))

        coeff1 = (matrix[0:halfDim, 0:halfDim] + matrix[halfDim:, halfDim:]) / 2
        coeffX = (matrix[halfDim:, 0:halfDim] + matrix[0:halfDim, halfDim:]) / 2
        coeffY = -I * (matrix[halfDim:, 0:halfDim] - matrix[0:halfDim, halfDim:]) / 2
        coeffZ = (matrix[0:halfDim, 0:halfDim] - matrix[halfDim:, halfDim:]) / 2

        coefficients = {"I": coeff1, "X": coeffX, "Y": coeffY, "Z": coeffZ}

        # Recursion for the Submatrices
        for c in coefficients:
            matrix = coefficients[c]
            if not (is_zero(matrix)):
                subDec = PauliDecomposition(matrix, f"{PauliStringInit}{c}")
                decomposition.update(subDec)

    return decomposition


def is_zero(matrix):
    if isinstance(matrix, sympy.Matrix):
        return matrix.is_zero
    elif sparse.issparse(matrix):
        return len(matrix.nonzero()[0]) == 0
    elif isinstance(matrix, np.ndarray):
        return not (matrix.any())
    else:
        raise ValueError("Unknown matrix type.")
