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
"""Tests for LinearOperatorBand."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import array_ops
from tensorflow.python.ops.linalg import linalg as linalg_lib
from tensorflow.python.ops.linalg import linear_operator_test_util
from tensorflow.python.platform import test

linalg = linalg_lib


class LinearOperatorBandSquareTest(
    linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
  """Most tests done in the base class LinearOperatorDerivedClassTest."""

  @property
  def _operator_build_infos(self):
    build_info = linear_operator_test_util.OperatorBuildInfo
    return [
        # build_info((0, 0), offset=0, bandwidth=1),
        build_info((3, 3), num_subdiags=1, num_superdiags=0),
        # build_info((1, 1), num_subdiags=0, num_superdiags=0),
        # build_info((1, 3, 3), offset=0, bandwidth=1),
        # build_info((1, 3, 3), offset=1, bandwidth=2),
        # build_info((1, 3, 3), offset=2, bandwidth=5),
        # build_info((3, 4, 4), offset=2, bandwidth=4),
        # build_info((2, 1, 4, 4), offset=2, bandwidth=4)]
    ]

  def _operator_and_matrix(self, build_info, dtype, use_placeholder):
    shape = list(build_info.shape)
    batch_shape = shape[:-2]
    num_rows = shape[-2]
    num_cols = shape[-1]

    diag_length = min(num_rows, num_cols)

    diag = linear_operator_test_util.random_sign_uniform(
          batch_shape + [diag_length],
          minval=1., maxval=2., dtype=dtype)

    if build_info.num_subdiags == 0:
      subdiags = None
    else:
      subdiags = linear_operator_test_util.random_sign_uniform(
          batch_shape + [diag_length, build_info.num_subdiags],
          minval=1., maxval=2., dtype=dtype)

    if build_info.num_superdiags == 0:
      superdiags = None
    else:
      superdiags = linear_operator_test_util.random_sign_uniform(
          batch_shape + [diag_length, build_info.num_superdiags],
          minval=1., maxval=2., dtype=dtype)

    if use_placeholder:
      diags = [
          (None if diag is None else array_ops.placeholder_with_default(diag, shape=None))
          for diag in diags
      ]

    operator = linalg.LinearOperatorBand(num_rows, num_cols, diag, subdiags, superdiags)
    matrix = operator.to_dense()

    return operator, matrix

  def test_to_dense_with_hard_coded_matrix(self):
    # Test to_dense() against a hard-coded example.
    diag = [1., 2., 3.]
    subdiag = [4., 5.]
    superdiag = [6., 7.]
    operator = linalg.LinearOperatorBand(3, 3, [superdiag, diag, subdiag], 1)
    matrix = operator.to_dense()
    self.assertAllEqual(matrix,
                        [[1., 6., 0.],
                         [4., 2., 7.],
                         [0., 5., 3.]])

  def test_is_x_flags(self):
    pass

  def test_foo_raises(self):
    pass


class LinearOperatorBandSquareSingularTest(
    linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
  """Most tests done in the base class LinearOperatorDerivedClassTest."""

  @property
  def _tests_to_skip(self):
    return ["solve", "solve_with_broadcast"]

  @property
  def _operator_build_infos(self):
    build_info = linear_operator_test_util.OperatorBuildInfo
    return [
        build_info((1, 3, 3), offset=1, bandwidth=1),
        build_info((1, 3, 3), offset=-1, bandwidth=1),
        build_info((3, 4, 4), offset=2, bandwidth=2),
        build_info((2, 1, 4, 4), offset=-2, bandwidth=1)]

  def _operator_and_matrix(self, build_info, dtype, use_placeholder):
    shape = list(build_info.shape)
    batch_shape = shape[:-2]
    num_rows = shape[-2]
    num_cols = shape[-1]

    diags = []
    for i in range(build_info.bandwidth):
      d_offset = i - build_info.offset
      diag_length = min(num_rows - d_offset, num_rows, num_cols,
                        num_cols + d_offset)
      diag = linear_operator_test_util.random_sign_uniform(
          batch_shape + [diag_length], minval=1., maxval=2., dtype=dtype)
      diags.append(diag)

    if use_placeholder:
      diags = [
          array_ops.placeholder_with_default(diag, shape=None) for diag in diags
      ]

    operator = linalg.LinearOperatorBand(num_rows, num_cols, diags,
                                         build_info.offset)
    matrix = operator.to_dense()

    return operator, matrix


class LinearOperatorBandNonSquareTest(
    linear_operator_test_util.NonSquareLinearOperatorDerivedClassTest):
  """Most tests done in the base class LinearOperatorDerivedClassTest."""

  @property
  def _operator_build_infos(self):
    build_info = linear_operator_test_util.OperatorBuildInfo
    return [
        build_info((2, 1), offset=0, bandwidth=2),
        build_info((1, 2), offset=1, bandwidth=2),
        build_info((1, 3, 2), offset=-1, bandwidth=2),
        build_info((3, 3, 4), offset=3, bandwidth=6),
        build_info((2, 1, 2, 4), offset=2, bandwidth=3)]

  def _operator_and_matrix(self, build_info, dtype, use_placeholder):
    shape = list(build_info.shape)
    batch_shape = shape[:-2]
    num_rows = shape[-2]
    num_cols = shape[-1]

    diags = []
    for i in range(build_info.bandwidth):
      d_offset = i - build_info.offset
      diag_length = min(num_rows - d_offset, num_rows, num_cols,
                        num_cols + d_offset)
      diag = linear_operator_test_util.random_sign_uniform(
          batch_shape + [diag_length], minval=1., maxval=2., dtype=dtype)
      diags.append(diag)

    if use_placeholder:
      diags = [
          array_ops.placeholder_with_default(diag, shape=None) for diag in diags
      ]

    operator = linalg.LinearOperatorBand(num_rows, num_cols, diags,
                                         build_info.offset)
    matrix = operator.to_dense()

    return operator, matrix

  def test_to_dense_with_hard_coded_matrix(self):
    # Test to_dense() against a hard-coded example.
    diag = [1., 2., 3.]
    subdiag = [4., 5., 6.]
    operator = linalg.LinearOperatorBand(4, 3, [diag, subdiag], offset=0)
    matrix = operator.to_dense()
    self.assertAllEqual(matrix,
                        [[1., 0., 0.],
                         [4., 2., 0.],
                         [0., 5., 3.],
                         [0., 0., 6.]])

  def test_is_x_flags(self):
    pass

  def test_foo_raises(self):
    pass


if __name__ == "__main__":
  test.main()
