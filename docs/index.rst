second_quantization
==================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   tutorial/index
   api

Converting second quantized operators to matrix form and back
------------------------------------------------------------

The ``second_quantization`` package provides tools for working with second quantized operators in quantum many-body systems. It allows you to:

* Convert second quantized expressions to matrix representations
* Work with different basis states and fermion/boson operators
* Analyze quantum many-body systems including quantum dots and superconducting devices
* Perform time evolution and spectroscopic calculations

Quick Start
-----------

Install the package and start exploring quantum many-body systems:

.. code-block:: python

   import second_quantization as sq

   # Create a Hilbert space with 2 orbitals
   hilbert = sq.HilbertSpace(n_orbitals=2)

   # Define creation and annihilation operators
   c_dag = hilbert.creation_operator(0, spin='up')
   c = hilbert.annihilation_operator(0, spin='up')

   # Convert to matrix form
   matrix = hilbert.to_matrix(c_dag * c)

Tutorials
---------

Check out our comprehensive tutorials:

* **Tutorial Overview**: :doc:`tutorial/index` - Introduction to the package capabilities

API Reference
-------------

For detailed API documentation, see the :doc:`api` reference.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
