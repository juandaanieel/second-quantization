# # Minimal examples
#
# ## Sums or operators are problematic

# +
from second_quantization import hilbert_space
from IPython.display import display
from sympy.physics.quantum.fermion import FermionOp
from sympy.physics.quantum import Dagger

lof = [FermionOp("c_0"), FermionOp("c_1")]
expr = lof[0] + 4 * Dagger(lof[1]) * lof[1]
mat = hilbert_space.to_matrix(expr, lof, sparse=False)
ops = hilbert_space.to_operators(mat[1], lof)
display(expr)
display(ops)
# -

# ## Normal ordering is not picked up on in certain cases

# +

lof = [FermionOp("c_0"), FermionOp("c_1"), FermionOp("c_2")]
expr = lof[0] * lof[2] * lof[1]
mat = hilbert_space.to_matrix(expr, lof, sparse=False)
ops = hilbert_space.to_operators(mat[1], lof)
display(expr)
display(ops)
# -

# Here the reason is actually straight forward (I explain it in the chat): The
# JW string of the unordered expression cannot distinguish the ordering of the
# operators because it looks all the same no matter what order the operators
# have. Hence, the matrix representation is the same. For reference, that is
# always $\sigma_- \otimes \sigma_- \otimes \sigma_-$
