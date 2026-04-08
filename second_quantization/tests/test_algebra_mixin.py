import pytest

from second_quantization.fock_core.fermion_sequence import FermionSequence


def build_sequence(oper_list, weights):
    return FermionSequence(oper_list, weights=weights, bypass_parse=True)


def test_add_and_subtract_merge_like_terms_and_identity():
    sequence = build_sequence([[0, 1]], [2])
    other = build_sequence([[0, 1], [2, 3]], [3, 4])

    added = sequence + other
    subtracted = sequence - 5

    assert added.oper_list == [[0, 1], [2, 3]]
    assert added.weights == [5, 4]

    assert subtracted.oper_list == [[0, 1], []]
    assert subtracted.weights == [2, -5]


def test_scalar_multiply_divide_and_negation_keep_operator_structure():
    sequence = build_sequence([[0, 1], [2, 3]], [2, -4])

    multiplied = sequence * 3
    reverse_multiplied = 3 * sequence
    divided = sequence / 2
    negated = -sequence

    assert multiplied.oper_list == sequence.oper_list
    assert multiplied.weights == [6, -12]

    assert reverse_multiplied.oper_list == sequence.oper_list
    assert reverse_multiplied.weights == [6, -12]

    assert divided.oper_list == sequence.oper_list
    assert divided.weights == [1, -2]

    assert negated.oper_list == sequence.oper_list
    assert negated.weights == [-2, 4]


def test_commutator_and_anticommutator_for_single_mode_match_fermion_algebra():
    annihilation = build_sequence([[1]], [1])
    creation = build_sequence([[0]], [1])

    commutator = annihilation.commute(creation)
    anticommutator = annihilation.anticommute(creation)

    assert commutator.oper_list == [[1, 0], []]
    assert commutator.weights == [-2, 1]

    assert anticommutator.oper_list == [[]]
    assert anticommutator.weights == [1]


def test_dagger_reverses_operator_order_and_conjugates_weights():
    sequence = build_sequence([[0, 3, 4], []], [1 + 2j, -3j])

    daggered = ~sequence

    assert daggered.oper_list == [[5, 2, 1], []]
    assert daggered.weights == [1 - 2j, 3j]


def test_shift_and_mod_operators_follow_sequence_lengths_and_sites():
    sequence = build_sequence([[0, 1], [2, 3]], [1, 2])

    shifted = sequence >> 1
    shifted_back = shifted << 1

    assert shifted.oper_list == [[4, 5], [6, 7]]
    assert shifted.weights == [1, 2]

    assert shifted_back.oper_list == sequence.oper_list
    assert shifted_back.weights == sequence.weights

    assert sequence % 2 is True
    assert build_sequence([[0], [2, 3]], [1, 1]) % 2 is False


def test_equality_is_insensitive_to_term_order():
    left = build_sequence([[0], [1], []], [1, 2, 3])
    right = build_sequence([[], [0], [1]], [3, 1, 2])

    assert left == right


def test_scalar_operations_reject_unsupported_types():
    sequence = build_sequence([[0, 1]], [1])

    with pytest.raises(TypeError):
        sequence + "x"

    with pytest.raises(TypeError):
        sequence * object()

    with pytest.raises(TypeError):
        sequence / "x"
