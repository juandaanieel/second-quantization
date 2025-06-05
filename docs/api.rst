API Reference
=============

This section contains the complete API documentation for the ``second_quantization`` package.

Package Overview
----------------

Core Modules
------------

.. currentmodule:: second_quantization

Hilbert Space Operations
~~~~~~~~~~~~~~~~~~~~~~~~

The :mod:`second_quantization.hilbert_space` module provides the core functionality for
converting between second-quantized operator expressions and matrix representations.

.. automodule:: second_quantization.hilbert_space
   :members:
   :undoc-members:
   :show-inheritance:

Main Functions
^^^^^^^^^^^^^^

.. autofunction:: second_quantization.hilbert_space.to_matrix
.. autofunction:: second_quantization.hilbert_space.to_operators
.. autofunction:: second_quantization.hilbert_space.make_dict_callable
.. autofunction:: second_quantization.hilbert_space.parity_operator

Basis Generation
^^^^^^^^^^^^^^^^

.. autofunction:: second_quantization.hilbert_space.basis_operators
.. autofunction:: second_quantization.hilbert_space.string_basis
.. autofunction:: second_quantization.hilbert_space.symbolic_basis
.. autofunction:: second_quantization.hilbert_space.find


Pauli String Decomposition
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :mod:`second_quantization.pauli_strings` module provides tools for decomposing
matrices into Pauli string representations.

.. automodule:: second_quantization.pauli_strings
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: second_quantization.pauli_strings.PauliDecomposition
.. autofunction:: second_quantization.pauli_strings.is_zero
