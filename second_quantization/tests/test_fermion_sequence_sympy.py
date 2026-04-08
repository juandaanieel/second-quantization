import numpy as np
import sympy
from sympy.physics.quantum import Dagger
from sympy.physics.quantum.fermion import FermionOp

from second_quantization.fock_core.fermion_sequence import FermionSequence


def _compile_array(expr, operators, basis_size=2):
    return (
        FermionSequence(expr, sympy_operators=operators)
        .compile(fock_basis=basis_size, data_format="array")
        .get()
    )


def test_compile_builds_param_dict_and_updates_relative_signs():
    t = sympy.Symbol("t")
    c_down = FermionOp("c_{0,down}")
    c_up = FermionOp("c_{0,up}")
    operators = [c_down, c_up]

    expr = t * (c_up * c_down - c_down * c_up)
    h_data = FermionSequence(expr, sympy_operators=operators).compile(
        fock_basis=2, data_format="array"
    )

    np.testing.assert_allclose(h_data.get(), np.zeros((4, 4), dtype=complex))

    h_data.update_values({t: 2})
    expected = _compile_array(expr.subs({t: 2}), operators)
    np.testing.assert_allclose(h_data.get(), expected)

    h_data.update_values({t: -3})
    expected = _compile_array(expr.subs({t: -3}), operators)
    np.testing.assert_allclose(h_data.get(), expected)


def test_compile_keeps_numeric_terms_and_updates_general_sympy_weights():
    x, y = sympy.symbols("x y")
    c_down = FermionOp("c_{0,down}")
    operators = [c_down]

    expr = 2 * Dagger(c_down) * c_down + (x**2 - y**2) * Dagger(c_down) * c_down
    h_data = FermionSequence(expr, sympy_operators=operators).compile(
        fock_basis=1, data_format="array"
    )

    expected_initial = _compile_array(
        (2 * Dagger(c_down) * c_down).expand(), operators, 1
    )
    np.testing.assert_allclose(h_data.get(), expected_initial)

    h_data.update_values({x: 3, y: 1})
    expected = _compile_array(expr.subs({x: 3, y: 1}), operators, 1)
    np.testing.assert_allclose(h_data.get(), expected)
