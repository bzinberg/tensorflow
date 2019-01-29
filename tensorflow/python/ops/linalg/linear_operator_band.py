# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""`LinearOperator` acting like a band matrix."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework.smart_cond import smart_cond
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linalg_impl as linalg
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.util.tf_export import tf_export

__all__ = [
    "LinearOperatorBand",
]


def _spread_columns_down(d, num_rows):
  # Transpose first and last dims of d so we can use tf.scan.
  rank = array_ops.rank(d)
  perm = array_ops.concat([[rank - 1], math_ops.range(1, rank - 1), [0]], axis=0)
  d = array_ops.transpose(d, perm=perm)

  shape = array_ops.shape(d)
  k = shape[-1]
  max_diag_length = shape[-2]
  batch_shape = shape[:-2]

  def zero_col(num):
    return array_ops.zeros(array_ops.concat([batch_shape, [num]], axis=0), dtype=d.dtype)

  def pad_column(tup):
    i, d_col = tup
    diag_length = math_ops.minimum(max_diag_length, num_rows - i)
    return array_ops.concat([
        zero_col(i),
        d_col[..., :diag_length],
        zero_col(num_rows - i - diag_length),
    ], axis=-1)

  # Un-transpose first and last dims.
  return array_ops.transpose(
      functional_ops.map_fn(pad_column, (math_ops.range(k), d), dtype=d.dtype), perm=perm)

def _diag_matmul(diag, num_rows, num_cols, x):
  def zero_rows(num):
    shape = array_ops.shape(x)
    batch_shape = shape[:-2]
    return tf.zeros(tf.concat([batch_shape, [num, num_cols]], axis=0),
                    dtype=x.dtype)
  return smart_cond(num_rows <= num_cols,
                    lambda: diag[..., array_ops.newaxis] * x[..., :num_rows],
                    lambda: array_ops.concat(
                        [diag[..., array_ops.newaxis] * x,
                         zero_rows(num_rows - num_cols)], axis=-2))

def _subdiags_matmul(subdiags, num_rows, num_cols, x, batch_shape=None):
  if batch_shape is None:
    batch_shape = []

  xtilde = subdiags[..., :, array_ops.newaxis, :] * x[..., 1:, :, array_ops
                                                      .newaxis]
  return array_ops.concat([
      array_ops.zeros(
          array_ops.concat([batch_shape, [1, num_cols]], axis=0),
          dtype=x.dtype),
      _splay_down(xtilde, num_rows),
  ], axis=-2)

def _superdiags_matmul(superdiags, num_rows, num_cols, x, batch_shape=None):
  if batch_shape is None:
    batch_shape = []

  xtilde = superdiags[..., array_ops.newaxis, :] * x[..., array_ops.newaxis]
  return _sum_desc_slices(xtilde, num_rows)

def _splay_down(x, num_rows):
  shape = array_ops.shape(x)
  k = shape[-1]
  batch_shape = shape[:-3]
  x_num_rows = shape[-3]
  x_num_cols = shape[-2]

  def zero_rows(num):
    return array_ops.zeros(
        array_ops.concat([batch_shape, [num, x_num_cols]], axis=0),
        dtype=x.dtype)

  def splay_slice(i, t):
    diag_length = math_ops.minimum(x_num_rows, height - i)
    return array_ops.concat([
        zero_rows(i),
        t[..., :diag_length, :],
        zero_rows(height - i - diag_length),
    ], axis=-2)

  i_final, t_final = control_flow_ops.while_loop(
      lambda i, t: i < k,
      lambda i, t: (i+1, t + splay_slice(i, x[..., i])),
      [0, zero_rows(height)])

  return t_final

def _sum_desc_slices(x, max_diag_length):
  shape = array_ops.shape(x)
  k = shape[-1]
  batch_shape = shape[:-3]
  x_num_rows = shape[-3]
  x_num_cols = shape[-2]

  def zero_rows(num):
    return array_ops.zeros(
        array_ops.concat([batch_shape, [num, x_num_cols]], axis=0),
        dtype=x.dtype)

  def get_slice(i):
    return control_flow_ops.cond(
        i <= x_num_rows - max_diag_length,
        lambda: x[..., i:i + max_diag_length, :, i - 1],
        lambda: array_ops.concat([x[..., i:x_num_rows, :, i - 1], zero_rows(max_diag_length - (x_num_rows - i))], axis=-2))

  i_final, t_final = control_flow_ops.while_loop(
      lambda i, t: i < k,
      lambda i, t: (i+1, t + get_slice(i)),
      [1, zero_rows(max_diag_length)])
  return t_final


def _diag_matmul(diag, d_offset, m, n, x):
  """Matmul by a (possibly non-square, off-center) diagonal matrix.

  Returns the matrix product `D @ x`, where `D` is a matrix of supplied
  dimensions `[m, n]` whose diagonal at `d_offset` spaces below the main
  diagonal has supplied elements `diag`.

  Args:
    diag: (Batch) vector-shaped `Tensor` representing the diagonal elements.
    d_offset: Python integer representing number of spaces below the main
      diagonal.
    m: Scalar integer `Tensor` representing number of rows.
    n: Scalar integer `Tensor` representing number of columns.
    x: (Batch) matrix-shaped `Tensor` with the same dtype as `diag`.

  Returns:
    product: (Batch) matrix-shaped `Tensor` representing the matrix product
      `D @ x`.
  """
  expanded_diag = diag[..., array_ops.newaxis]
  if x.shape.ndims is not None and x.shape[-1].value is not None:
    x_num_cols = x.shape[-1].value
  else:
    x_num_cols = array_ops.shape(x)[-1]

  if diag.shape.is_fully_defined():
    batch_shape = diag.shape.as_list()[:-1]
  else:
    batch_shape = array_ops.shape(diag)[:-1]

  # Helper function to create `num_rows` rows of zeros
  def zero_rows(num_rows):
    return array_ops.zeros(array_ops.concat([
        batch_shape,
        ops.convert_to_tensor([num_rows, x_num_cols]),
    ], axis=0), diag.dtype)

  # Draw a line through the diagonal. Which two edges of the matrix does the
  # line cross? Four different cases, depending on the answer.
  def left_and_bottom_edge():
    return array_ops.concat([
        zero_rows(d_offset),
        expanded_diag * x[..., :m - d_offset, :],
    ], axis=-2)

  def left_and_right_edge():
    return array_ops.concat([
        zero_rows(d_offset),
        expanded_diag * x,
        zero_rows(m - n - d_offset),
    ], axis=-2)

  def top_and_bottom_edge():
    return expanded_diag * x[..., -d_offset:m - d_offset, :]

  def top_and_right_edge():
    return array_ops.concat([
        expanded_diag * x[..., -d_offset:n, :],
        zero_rows(m - n - d_offset),
    ], axis=-2)

  if d_offset >= 0:
    return smart_cond(m - d_offset <= n,
                      left_and_bottom_edge,
                      left_and_right_edge)
  else:
    return smart_cond(m - d_offset <= n,
                      top_and_bottom_edge,
                      top_and_right_edge)


def _make_diag_at_offset(diag, d_offset, m, n):
  """Returns a (possibly non-square, off-center) diagonal matrix.

  Args:
    diag: (Batch) vector-shaped `Tensor` representing the diagonal elements.
    d_offset: Python integer representing number of spaces below the main
      diagonal.
    m: Scalar integer `Tensor` representing number of rows.
    n: Scalar integer `Tensor` representing umber of columns.

  Returns:
    diag_mat: (Batch of) matrix-shaped `Tensor` representing the `m` by `n`
      matrix whose diagonal at `d_offset` spaces below the main diagonal has
      elements `diag`.
  """
  if diag.shape.is_fully_defined():
    batch_shape = diag.shape.as_list()[:-1]
  else:
    batch_shape = array_ops.shape(diag)[:-1]

  def zero_rows(num_rows, row_length):
    return array_ops.zeros(array_ops.concat([
        batch_shape,
        ops.convert_to_tensor([num_rows, row_length]),
    ], axis=0), diag.dtype)

  def zero_cols(num_cols, col_length):
    return array_ops.zeros(array_ops.concat([
        batch_shape,
        ops.convert_to_tensor([col_length, num_cols]),
    ], axis=0), diag.dtype)

  def left_and_bottom_edge():
    return array_ops.concat([
        zero_rows(d_offset, n),
        array_ops.concat([
            linalg.diag(diag),
            zero_cols(n - (m - d_offset), m - d_offset),
        ], axis=-1),
    ], axis=-2)

  def left_and_right_edge():
    return array_ops.concat([
        zero_rows(d_offset, n),
        linalg.diag(diag),
        zero_rows(m - n - d_offset, n),
    ], axis=-2)

  def top_and_bottom_edge():
    return array_ops.concat([
        zero_cols(-d_offset, m),
        linalg.diag(diag),
        zero_cols(n - m + d_offset, m),
    ], axis=-1)

  def top_and_right_edge():
    return array_ops.concat([
        zero_cols(-d_offset, m),
        array_ops.concat([
            linalg.diag(diag),
            zero_rows(m - (n + d_offset), n + d_offset),
        ], axis=-2),
    ], axis=-1)

  if d_offset >= 0:
    return smart_cond(m - d_offset <= n,
                      left_and_bottom_edge,
                      left_and_right_edge)
  else:
    return smart_cond(m - d_offset <= n,
                      top_and_bottom_edge,
                      top_and_right_edge)


@tf_export("linalg.LinearOperatorBand")
class LinearOperatorBand(linear_operator.LinearOperator):
  """`LinearOperator` acting like a [batch] band matrix.

  This operator acts like a [batch] band matrix `A`.  The band is specified by a
  nonempty list of [batch] vectors `diags` containing the diagonals, in
  descending order starting from `offset` spaces above the main diagonal.

  Has shape `[B1,...,Bb, M, N]` for some `b >= 0` and supplied `M, N`.  The
  first `b` indices index a batch member.  For every batch index `(i1,...,ib)`,
  `A[i1,...,ib, : :]` is an `M x N` matrix.  This matrix `A` is not
  materialized, but for purposes of broadcasting this shape will be relevant.

  `diags[i]` is a [batch] vector representing the diagonal `i - offset` spaces
  below the main diagonal (e.g., `i = offset` corresponds to the main diagonal;
  `i = offset - 1` corresponds to the superdiagonal).  Therefore `diags[i]` must
  have shape `[B1,...,Bb, D(i - offset)]`, where `D(j)` is the number of
  elements of the diagonal `j` spaces below the main diagonal:

  ```none
  D(j) = min(M - j, M, N, N + j)
  ```

  ```python
  # Create a 2 x 2 upper bidiagonal linear operator.
  superdiag = [2.]
  diag = [1., -1.]
  operator = LinearOperatorBand(2, 2, [superdiag, diag], offset=1)

  operator.to_dense()
  ==> [[1.,  2.]
       [0., -1.]]

  operator.shape
  ==> [2, 2]

  operator.log_abs_determinant()
  ==> scalar Tensor

  x = ... Shape [2, 4] Tensor
  operator.matmul(x)
  ==> Shape [2, 4] Tensor

  # Create a 5 x 3 linear operator with band entirely below the diagonal.
  subdiag = [1., 2., 3.]
  subsubdiag = [4., 5., 6.]
  operator = LinearOperatorBand(5, 3, [subdiag, subsubdiag], offset=-1)

  operator.to_dense()
  ==> [[0., 0., 0.]
       [1., 0., 0.]
       [4., 2., 0.]
       [0., 5., 3.]
       [0., 0., 6.]]

  # Create a [2, 3] batch of 4 x 4 tridiagonal linear operators.
  superdiag = tf.random_normal(shape=[2, 3, 3])
  diag = tf.random_normal(shape=[2, 3, 4])
  subdiag = tf.random_normal(shape=[2, 3, 3])
  operator = LinearOperatorBand(4, 4, [superdiag, diag, subdiag], offset=1)

  # Create a shape [2, 1, 4, 2] vector.  Note that this shape is compatible
  # since the batch dimensions, [2, 1], are broadcast to
  # operator.batch_shape = [2, 3].
  y = tf.random_normal(shape=[2, 1, 4, 2])
  x = operator.solve(y)
  ==> operator.matmul(x) = y
  ```

  #### Shape compatibility

  This operator acts on [batch] matrix with compatible shape.
  `x` is a batch matrix with compatible shape for `matmul` and `solve` if

  ```
  operator.shape = [B1,...,Bb] + [M, N],  with b >= 0
  x.shape =   [C1,...,Cc] + [N, R],
  and [C1,...,Cc] broadcasts with [B1,...,Bb] to [D1,...,Dd]
  ```

  #### Performance

  Suppose `operator` is a `LinearOperatorBand` of shape `[M, N]`, with diagonals
  of length `d1,...,dK` where `K = len(diags)`, and `x.shape = [N, R]`.  Let
  `dTotal = d1 + ... + dK`.  Then

  * `operator.matmul(x)` involves `dTotal * R` multiplications and `K` `N x R`
    matrix additions.
  * `operator.solve(x)` attempts to perform the following specialized algorithms
     in order of preference: diagonal solve, triangular solve, generic solve.
  * `operator.determinant()` involves a size `N` `reduce_prod` if the operator
    is triangular, otherwise we fall back to generic `O(N^3)` determinant.

  If instead `operator` and `x` have shape `[B1,...,Bb, N, N]` and
  `[B1,...,Bb, N, R]`, every operation increases in complexity by `B1*...*Bb`.

  #### Matrix property hints

  This `LinearOperator` is initialized with boolean flags of the form `is_X`,
  for `X = non_singular, self_adjoint, positive_definite, square`.
  These have the following meaning:

  * If `is_X == True`, callers should expect the operator to have the
    property `X`.  This is a promise that should be fulfilled, but is *not* a
    runtime assert.  For example, finite floating point precision may result
    in these promises being violated.
  * If `is_X == False`, callers should expect the operator to not have `X`.
  * If `is_X == None` (the default), callers should have no expectation either
    way.
  """

  def __init__(self,
               num_rows,
               num_cols,
               main_diag,
               subdiags=None,
               superdiags=None,
               is_non_singular=None,
               is_self_adjoint=None,
               is_positive_definite=None,
               is_square=None,
               name="LinearOperatorBand"):
    r"""Initialize a `LinearOperatorBand`.

    Args:
      num_rows: Number of rows, aka, dimension of the range.  Type must be
        either `int`-like, or scalar integer `Tensor`.
      num_cols: Number of columns, aka, dimension of the domain.  Type must be
        either `int`-like, or scalar integer `Tensor`.
      main_diag: (Batch) vector-shaped `Tensor` representing the main diagonal.
      subdiags: (Batch) matrix-shaped `Tensor` whose `j`th column represents the
        subdiagonal `j + 1` spaces below the main diagonal.  Has shape
        `[min(num_rows, num_cols), k]`, where `k` is the number of subdiagonals.
        Subdiagonals that have fewer than `min(num_rows, num_cols)` elements
        (due to being cut off by the bottom edge of the matrix) should be
        positioned flush against the top of `subdiags`.
        Allowed dtypes: `float16`, `float32`, `float64`, `complex64`,
        `complex128`.
        Default value: `None`, meaning no subdiagonals.
      superdiags: (Batch) matrix-shaped `Tensor` whose `j`th column represents
        the superdiagonal `j + 1` spaces above the main diagonal.
        Superdiagonals that have fewer than `min(num_rows, num_cols)` elements
        (due to being cut off by the right edge of the matrix) should be
        positioned flush against the top of `superdiags`.
        Allowed dtypes: `float16`, `float32`, `float64`, `complex64`,
        `complex128`.
        Default value: `None`, meaning no superdiagonals.
      is_non_singular:  Expect that this operator is non-singular.
      is_self_adjoint:  Expect that this operator is equal to its hermitian
        transpose.
      is_positive_definite:  Expect that this operator is positive definite,
        meaning the quadratic form `x^H A x` has positive real part for all
        nonzero `x`.  Note that we do not require the operator to be
        self-adjoint to be positive-definite.  See:
        https://en.wikipedia.org/wiki/Positive-definite_matrix#Extension_for_non-symmetric_matrices
      is_square:  Expect that this operator acts like square [batch] matrices.
      name: A name for this `LinearOperator`.

    Raises:
      TypeError:  If an element of `diags` has a non-allowed type.
      ValueError:  If an element of `diags` is statically known to have the
        wrong shape.
      ValueError:  If the specified band is too wide to fit inside the matrix.
      ValueError:  If `is_self_adjoint` is `True` but the band does not extend
        the same number of spaces on both sides of the diagonal.
    """
    with ops.name_scope(name, values=[num_rows, num_cols, main_diag, superdiags, subdiags]):
      self._main_diag = ops.convert_to_tensor(main_diag)
      # self._precomputed_batch_shape = self._check_diags_and_deduce_batch_shape(
      #     self._diags, self._offset)

      self._num_rows = None if isinstance(num_rows, ops.Tensor) else num_rows
      self._num_cols = None if isinstance(num_cols, ops.Tensor) else num_cols
      self._num_rows_tensor = ops.convert_to_tensor(num_rows)
      self._num_cols_tensor = ops.convert_to_tensor(num_cols)

      self._subdiags = None if subdiags is None else ops.convert_to_tensor(
          subdiags)
      # The alignment most convenient for matmul is, e.g., for a 3 by 4
      # matrix with superdiagonals [1,2,3], [4,5,6], [7,8]:
      #
      # [[1, 0, 0],
      #  [2, 4, 0],
      #  [3, 5, 7],
      #  [0, 6, 8]]
      self._superdiags = None if superdiags is None else _spread_columns_down(
          ops.convert_to_tensor(superdiags), num_cols)

      # if (self._num_rows is not None and
      #     -self._offset + len(self._diags) > self._num_rows):
      #   raise ValueError("Band extends past the bottom-left corner of matrix")
      # if (self._num_cols is not None and self._offset <= -self._num_cols):
      #   raise ValueError("Band extends past the top-right corner of matrix")

      # if (is_square and
      #     self._num_rows is not None and
      #     self._num_cols is not None and
      #     self._num_rows != self._num_cols):
      #   raise ValueError(
      #       "is_square is True but num_rows != num_cols ({} vs. {}).".format(
      #           self._num_rows, self._num_cols))

      # if is_self_adjoint and len(self._diags) != 2 * self._offset + 1:
      #   raise ValueError(
      #       "Self-adjoint band matrix should have bandwidth 2 * offset + 1")

      super(LinearOperatorBand, self).__init__(
          dtype=self._main_diag.dtype,
          graph_parents=[
              t for t in [
                  self._main_diag, self._subdiags, self._superdiags, self
                  ._num_rows_tensor, self._num_cols_tensor
              ] if t is not None
          ],
          is_non_singular=is_non_singular,
          is_self_adjoint=is_self_adjoint,
          is_positive_definite=is_positive_definite,
          is_square=is_square,
          name=name)

  def _check_diags_and_deduce_batch_shape(self, diags, offset):
    """Static check of diagonals, and deduces the batch shape.

    Args:
      diags: List whose `i`-th element is a `Tensor` representing the (batch)
        diagonal `i - offset` spaces below the main diagonal.
      offset: Python integer indicating the location of the upper limit of the
        band.

    Returns:
      batch_shape: `TensorShape` indicating the most specific batch shape that
        can be deduced from the (static) batch shapes of the diagonals.

    Raises:
      ValueError: if `diags` is empty.
      TypeError: if two elements of `diags` have different dtypes
      TypeError: if the dtype of elements of `diags` is not an allowed type.
      ValueError: if two elements of `diags` have (statically) incompatible
        batch shapes.
      ValueError: if an element of `diags` has (statically known) rank 0.
    """
    if not diags:
      # Diags can't be empty because then we can't infer batch shape.
      raise ValueError("diags cannot be empty")

    allowed_dtypes = [
        dtypes.float16,
        dtypes.float32,
        dtypes.float64,
        dtypes.complex64,
        dtypes.complex128,
    ]

    dtype = diags[0].dtype
    if dtype not in allowed_dtypes:
      raise TypeError(
          "Diagonal dtype {} is not an allowed type. Must be one of: {}".format(
              dtype, allowed_dtypes))

    for i, diag in enumerate(diags):
      if diag.dtype != dtype:
        raise TypeError(
            "Diags 0 and {} have different dtypes: {} and {}".format(
                i, dtype, diag.dtype))

    batch_shape = None
    for i, diag in enumerate(diags):
      if diag.shape.ndims == 0:
        raise ValueError("Diag {} has rank 0, which is not allowed".format(i))
      if not diag.shape[:-1].is_compatible_with(batch_shape):
        raise ValueError((
            "Diag {} has batch shape incompatible with the batch shape implied"
            " by previous diags: {} vs. {}"
        ).format(i, diag.shape[:-1], batch_shape))
      batch_shape = diag.shape[:-1].merge_with(batch_shape)

    return batch_shape

  def _shape(self):
    batch_shape = self._precomputed_batch_shape
    return batch_shape.concatenate([self._num_rows, self._num_cols])

  def _shape_tensor(self):
    d_shape = array_ops.shape(self._diags[0])
    return array_ops.concat([
        d_shape[:-1],
        [self._num_rows_tensor, self._num_cols_tensor],
    ], axis=0)

  def _assert_non_singular(self):
    if self._is_lower_triangular or self._is_upper_triangular:
      has_main_diag = check_ops.assert_equal(
          True, self._main_diag is not None)
      with ops.control_dependencies([has_main_diag]):
        return linear_operator_util.assert_no_entries_with_modulus_zero(
            self._main_diag,
            message="Singular operator: Triangular and diagonal contains a"
            " zero.")
    return super(LinearOperatorBand, self)._assert_non_singular()

  def _assert_positive_definite(self):
    # TODO(b/118783630): Use specialized Hermitian eigensolver when matrix is
    # bidiagonal.
    hermitian_part = 0.5 * (
        self.to_dense() + linalg.adjoint(self.to_dense()))
    eigvals = linalg.eigvalsh(hermitian_part)
    # eigvals is real, so this is effective a cast
    eigvals = math_ops.real(eigvals)
    return check_ops.assert_positive(
        eigvals,
        message=("This operator is not positive-definite: its Hermitian part "
                 "has a non-positive eigenvalue."))

  def _assert_self_adjoint(self):
    square = check_ops.assert_equal(
        self._num_rows,
        self._num_cols,
        message="This operator is not square, hence not self-adjoint:"
        " num_rows != num_cols")
    with ops.control_dependencies([square]):
      # TODO(bzinberg): Fix this to allow different number of diags as long
      # as the extra diags are all zero.
      correct_num_diags = check_ops.assert_equal(
          len(self._diags), 2 * self._offset + 1, message="...")
      with ops.control_dependencies([correct_num_diags]):
        halfway = (len(self._diags) + 1) / 2
        with ops.control_dependencies([
            check_ops.assert_equal(a, math_ops.conj(b),
                                   message="...")
            for a, b in zip(self._diags[:halfway], self._diags[halfway::-1])
        ]):
          check_done = ops.no_op()
        return check_done

  def _matmul(self, x, adjoint=False, adjoint_arg=False):
    x = linalg.adjoint(x) if adjoint_arg else x
    n = self.domain_dimension_tensor()
    m = self.range_dimension_tensor()
    if adjoint:
      product = _diag_matmul(math_ops.conj(self._main_diag),
                             n, m, x)
      if self._subdiags is not None:
        product += _superdiags_matmul(linalg.adjoint(self._subdiags), n, m, x)
      if self._superdiags is not None:
        product += _subdiags_matmul(linalg.adjoint(self._superdiags), n, m, x)
    else:
      product = _diag_matmul(self._main_diag, m, n, x)
      if self._subdiags is not None:
        product += _superdiags_matmul(self._subdiags, m, n, x)
      if self._superdiags is not None:
        product += _subdiags_matmul(self._superdiags, m, n, x)
    return product

  def _determinant(self):
    if self._is_lower_triangular or self._is_upper_triangular:
      return math_ops.reduce_prod(self._main_diag, reduction_indices=[-1])
    return super(LinearOperatorBand, self)._determinant()

  def _log_abs_determinant(self):
    if self._is_lower_triangular or self._is_upper_triangular:
      log_det = math_ops.reduce_sum(
          math_ops.log(math_ops.abs(self._main_diag)), reduction_indices=[-1])
      if self.dtype.is_complex:
        log_det = math_ops.cast(log_det, dtype=self.dtype)
      return log_det
    return super(LinearOperatorBand, self)._log_abs_determinant()

  def _solve(self, rhs, adjoint=False, adjoint_arg=False):
    # TODO(b/118783765): Use a specialized bidiagonal solver if matrix is
    # bidiagonal.
    adjusted_rhs = linalg.adjoint(rhs) if adjoint_arg else rhs
    if self._is_diagonal:
      diag_term = math_ops.conj(
          self.diag_part()) if adjoint else self.diag_part()
      return adjusted_rhs / diag_term[..., array_ops.newaxis]
    if self._is_lower_triangular:
      return linear_operator_util.matrix_triangular_solve_with_broadcast(
          self._to_dense(), adjusted_rhs, lower=True, adjoint=adjoint)
    if self._is_upper_triangular:
      return linear_operator_util.matrix_triangular_solve_with_broadcast(
          self._to_dense(), adjusted_rhs, lower=False, adjoint=adjoint)
    return super(LinearOperatorBand, self)._solve(rhs, adjoint, adjoint_arg)

  def _to_dense(self):
    return control_flow_ops.while_loop(.....)
    # return sum(
    #     _make_diag_at_offset(diag, i - self._offset,
    #                          self.range_dimension_tensor(),
    #                          self.domain_dimension_tensor())
    #     for i, diag in enumerate(self._diags))

  def _diag_part(self):
    return self._main_diag

  @property
  def _is_diagonal(self):
    return self._subdiags is None and self._superdiags is None

  @property
  def _is_lower_triangular(self):
    return self._superdiags is None

  @property
  def _is_upper_triangular(self):
    return self._subdiags is None
