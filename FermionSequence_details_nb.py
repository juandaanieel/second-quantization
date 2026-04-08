import numpy as np
import sympy as sp
from sympy.physics.quantum import Dagger
from sympy.physics.quantum.fermion import FermionOp

from second_quantization import fock_space

"""
# FermionSequence internals

This notebook gives a compact walkthrough of how a `FermionSequence` is compiled
into the data representation.

This involves the following three functions:

1. `compile()` (in Mixins/compile.py)
2. `act_oper_list()` (in base.py)
3. `act_oper()` (in base.py)

The goal is to explain how symbolic operator strings are turned into rows,
columns, and values of a matrix representation in a Fock basis.
"""

"""
## Global oberview

A `FermionSequence` stores:

- `oper_list`: a list of operator subsequences
- `weights`: one prefactor per subsequence

Each operator is encoded as an integer:

- the lowest bit says annihilation (`1`) or creation (`0`)
- the remaining bits encode the fermionic mode index

A full Hamiltonian is then the sum of terms terms:

`weight_i * oper_list_i`

and compilation means:

- apply each operator subsequence to all states in a given Fock basis
- keep only the surviving actions
- store the resulting transitions as sparse-style data
"""

###############################################################################
# ## Small example system

c_down = FermionOp("c_{0,\\downarrow}")
c_up = FermionOp("c_{0,\\uparrow}")
t, U, mu = sp.symbols("t U mu")

H_sympy = (
    -t * Dagger(c_down) * c_up
    - t * Dagger(c_up) * c_down
    + U * Dagger(c_down) * c_down * Dagger(c_up) * c_up
    - mu * Dagger(c_down) * c_down
    - mu * Dagger(c_up) * c_up
)

ops = [c_down, c_up]
H_seq = fock_space.FermionSequence(H_sympy, sympy_operators=ops)

H_seq

"""
## What `compile()` does

At the highest level, `compile()` in
`fock_core/Mixins/compiler.py`

- validates the requested output format
- builds or validates a `FockBasis`
- dispatches to either:
  - `_construct_data_repr_ordered(...)`
  - `_construct_data_repr_restricted(...)`
- wraps the result in `FermionData`

The main branch point is whether the basis is ordered:

- ordered basis:
  matrix row/column indices are the state integers themselves
- restricted basis:
  states first need to be mapped back to basis indices

There is also a second branch:

- `cached=False`: apply every operator list independently
- `cached=True`: reuse partial results for terms that share a common prefix (needs testing still, but generally x2 faster. May not be worth the extra complexity)
"""

###############################################################################
# ## Calling compile

H_data = H_seq.compile(2, data_format="coo", cached=False)
H_data

"""
## Data generated during compile

The ordered compile path constructs arrays with one entry per nonzero matrix
transition:

- `rows`: bra-state indices
- `cols`: ket-state indices
- `values`: numerical matrix elements already including the term weights
- `signs`: fermionic signs from operator application
- `type_dict`: maps `tuple(oper_list)` to the slice in those arrays
- `sympy_metadata`: update metadata for symbolic prefactors

The actual heavy work is repeatedly applying operator subsequences to basis
states. That is delegated to `act_oper_list()`, which itself repeatedly calls
`act_oper()`.
"""

"""
## `act_oper_list()`: applying a full subsequence

`act_oper_list(oper_list, states)` takes:

- one subsequence of encoded fermionic operators
- an array of basis states

and applies the operators one by one.

For every operator in the subsequence it keeps track of:

- which states survive
- what the new states are after the bit flip
- what fermionic sign is generated at that step

The cumulative sign of the whole subsequence is the product of the signs from
each individual operator action, after filtering out destroyed states.

So the role of `act_oper_list()` is:

- chain together several single-operator actions
- keep only the states that survive all of them
- return:
  - the original surviving states
  - the final transformed states
  - the net fermionic sign per surviving transition
"""

###############################################################################
# ## Example: a full operator subsequence

states = np.arange(4, dtype=np.int32)
example_term = H_seq.oper_list[0]  ## The charging term

survived_states_in, survived_states_out, signs = H_seq.act_oper_list(
    example_term, states
)

survived_states_in, survived_states_out, signs

###############################################################################
# The above results tells us that state '3' is mapped to states '3' with sign '1'.
#
# In readable terms:
#
#     The doubly occupied state (in binary: 0b11 = 3) is mapped to itself by the U term. All other terms are destroyed.

"""
## `act_oper()`: applying one fermionic operator

This is the key low-level function in the compile path.

Given:

- one encoded operator `oper`
- an array of Fock states `states`

it does four things:

1. decode the operator
2. determine which states survive
3. compute the fermionic sign
4. flip the occupation bit to produce the new states

More explicitly:

### 1. Decode the operator

The operator integer stores:

- action type in the lowest bit
- mode index in the higher bits

This gives:

- `check_bit`: whether the mode must be occupied/unoccupied to survive
- `flip_bit_pos`: which bit in the state integer is acted on
- `flip_bit`: the actual bitmask for flipping that occupation

### 2. Filter surviving states

A creation operator kills states where the target mode is already occupied.
An annihilation operator kills states where the target mode is empty.

This is implemented by checking the occupation of the acted-on bit and building
`states_alive_filt`.

### 3. Compute the fermionic sign

The sign comes from the parity of occupied modes at lower positions than the
acted-on mode.

The current implementation computes:

- mask out all lower bits (retain only those below position acting on)
- count the number of occupied lower bits with `np.bitwise_count`
- reduce modulo 2
- map even/odd parity to `+1/-1`

### 4. Flip the bit

Finally the target occupation is toggled by XOR with `flip_bit`.

So `act_oper()` returns:

- `states_alive_filt`
- `new_states`
- `signs`
"""

###############################################################################
# ## Example: a single operator action

single_oper = example_term[0]
alive_filter, new_states, single_signs = H_seq.act_oper(single_oper, states)

single_oper, alive_filter, new_states, single_signs

"""
## Why compile is fastest in the ordered basis

The ordered basis case avoids an important lookup step:

- the integer representing a Fock state is already its basis index

That means after `act_oper_list()` returns `(old_states, new_states, signs)`,
compile can directly write:

- `rows = old_states`
- `cols = new_states`

without any additional hash lookup or index conversion.

This is why `_construct_data_repr_ordered(...)` is the most direct compile path.
"""

"""
## Cached compilation

When `cached=True`, compile sorts the operator subsequences into a cache-friendly
order and reuses partial applications for neighboring terms that share a common
prefix.

The idea is:

- if two terms start with the same first few operators,
  those first few actions do not need to be recomputed from scratch
  E.g.: We can sort such that the operator sequences [0,1] and [0,1,4,5] are right after each other, then cache the intermediate result after [0,1] and start the second sequence at 4.

This optimization is implemented through:

- `_act_oper_list_cached(...)`

The cached path stores:

- the original surviving states
- the intermediate transformed states
- the accumulated signs

up to the shared prefix length, and then continues only from the first operator
where the terms differ.
"""

###############################################################################
# ## Cached vs uncached compile

# %timeit H_data_uncached = H_seq.compile( data_format="coo", cached=False) ## For small arrays difference will be negligible of course
# %timeit H_data_cached = H_seq.compile( data_format="coo", cached=True)


"""
## Symbolic prefactors during compile

If the term weights contain SymPy expressions, compile also constructs
symbolic update metadata.

This metadata is built before entering the main compile loops and stores:

- the updateable symbols
- which compiled symbolic contributions depend on which symbol
- the coefficient expression associated with each symbolic contribution

The important point is that this metadata is attached to compiled slices, not to
stringified operator lists from parsing.

That makes the update path robust with respect to:

- relative minus signs, such as `t * (A - B)`
- non-normal-ordered input
- later changes in the operator ordering used internally
This structure also allows for general pre-factors that are not just linear
"""

H_data = H_seq.compile(data_format="array", cached=False)
H_data._backend.sympy_metadata

###############################################################################
# ## Updating symbolic values after compile

H_data = H_seq.compile(data_format="array", cached=False)
H_data.update_values({t: 1.0, U: 2.0, mu: 0.5})
H_data.get()

"""
## Summary

The compile pipeline can be viewed as:

1. take every term in the `FermionSequence`
2. apply its operator subsequence to all basis states
3. keep only the surviving transitions
4. multiply by the fermionic sign and the term weight
5. store the result as sparse-style matrix data

Operationally:

- `compile()` organizes the workflow and output format
- `act_oper_list()` applies a term operator by operator
- `act_oper()` is the low-level bit-manipulation kernel

In practice, most runtime is spent in repeated calls to `act_oper()`, since that
is where the survival test, fermionic sign, and state update all happen.
"""
