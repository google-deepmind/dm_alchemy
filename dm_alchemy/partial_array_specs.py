# Lint as: python3
# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Define versions of dm_env specs with some dimensions of shape unknown."""

from dm_env import specs
import numpy as np

_INVALID_DTYPE = 'Expected dtype %r but found %r'
_INVALID_SHAPE_LEN = 'Shape %r has different length to spec %r'
_INVALID_SHAPE = 'Shape %r does not conform to spec %r'


class PartialArray(specs.Array):
  """An `Array` spec with optionally an unknown size on one dimension."""

  def __init__(self, shape, dtype, name=None):
    """Initializes a new `PartialArray` spec.

    Args:
      shape: An iterable specifying the array shape with up to 1 dimension with
        unknown size specified by -1.
      dtype: numpy dtype or string specifying the array dtype.
      name: Optional string containing a semantic name for the corresponding
        array. Defaults to `None`.

    Raises:
      ValueError: If there is more than 1 dimension with unknown size or a
        dimension with value < -1.
      TypeError: If `shape` is not an iterable of elements convertible to int,
        or if `dtype` is not convertible to a numpy dtype.
    """
    if any(size < -1 for size in shape):
      raise ValueError('No entry in shape may be < -1, shape is {}'.format(
          shape))
    if sum(size == -1 for size in shape) > 1:
      raise ValueError('Only 1 entry in shape may be -1, shape is {}'.format(
          shape))
    super().__init__(shape, dtype, name)

  def _validate_shape(self, shape):
    if len(shape) != len(self.shape):
      self._fail_validation(_INVALID_SHAPE_LEN, shape, self.shape)
    for array_size, spec_size in zip(shape, self.shape):
      if spec_size == -1:
        continue
      if array_size != spec_size:
        self._fail_validation(_INVALID_SHAPE, shape, self.shape)

  def validate(self, value):
    """Checks if value conforms to this spec.

    Args:
      value: a numpy array or value convertible to one via `np.asarray`.

    Returns:
      value, converted if necessary to a numpy array.

    Raises:
      ValueError: if value doesn't conform to this spec.
    """
    value = np.asarray(value)
    self._validate_shape(value.shape)
    if value.dtype != self.dtype:
      self._fail_validation(_INVALID_DTYPE, self.dtype, value.dtype)
    return value

  def generate_value(self):
    """Generate a test value which conforms to this spec.

    If the size is -1 on a dimension we can use any positive value, here we
    use 1.

    Returns:
      Test value.
    """
    example_shape = tuple(1 if size == -1 else size for size in self.shape)
    return np.zeros(shape=example_shape, dtype=self.dtype)

  def __repr__(self):
    return 'PartialArray(shape={}, dtype={}, name={})'.format(
        self.shape, repr(self.dtype), repr(self.name))

  def __reduce__(self):
    return PartialArray, (self._shape, self._dtype, self._name)
