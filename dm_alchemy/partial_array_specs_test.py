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
"""Tests for partial array specs."""
import pickle

from absl.testing import absltest
from absl.testing import parameterized
from dm_alchemy import partial_array_specs
import numpy as np


class PartialArrayTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(spec_shape=(1, 2), is_valid=True),
      dict(spec_shape=(-1, 2), is_valid=True),
      dict(spec_shape=(1, -2), is_valid=False),
      dict(spec_shape=(-1, -1), is_valid=False),
  )
  def testShapeValueError(self, spec_shape, is_valid):
    if is_valid:
      partial_array_specs.PartialArray(spec_shape, np.int32)
    else:
      with self.assertRaises(ValueError):
        partial_array_specs.PartialArray(spec_shape, np.int32)

  @parameterized.parameters(
      dict(value=np.zeros((1, 2), dtype=np.int32), is_valid=True),
      dict(value=np.zeros((2, 2), dtype=np.int32), is_valid=True),
      dict(value=np.zeros((2, 3), dtype=np.int32), is_valid=False),
      dict(value=np.zeros((1, 2, 3), dtype=np.int32), is_valid=False,
           error_format=partial_array_specs._INVALID_SHAPE_LEN),
  )
  def testValidateShape(
      self, value, is_valid, error_format=partial_array_specs._INVALID_SHAPE):
    spec = partial_array_specs.PartialArray((-1, 2), np.int32)
    if is_valid:  # Should not raise any exception.
      spec.validate(value)
    else:
      with self.assertRaisesWithLiteralMatch(
          ValueError, error_format % (value.shape, spec.shape)):
        spec.validate(value)

  def testGenerateValue(self):
    spec = partial_array_specs.PartialArray((2, -1), np.int32)
    test_value = spec.generate_value()
    spec.validate(test_value)

  def testSerialization(self):
    desc = partial_array_specs.PartialArray([-1, 5], np.float32, "test")
    self.assertEqual(pickle.loads(pickle.dumps(desc)), desc)


if __name__ == "__main__":
  absltest.main()
