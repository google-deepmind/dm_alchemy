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
"""Tests for dm_alchemy.load_from_disk."""

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import dm_alchemy
from dm_env import test_utils

FLAGS = flags.FLAGS

flags.DEFINE_string('path', '',
                    'Directory that contains dm_alchemy environment.')

_TEST_LEVEL = ('alchemy/perceptual_mapping_randomized_with_rotation_and_random'
               '_bottleneck')

_TEST_LEVELS = (
    'alchemy/perceptual_mapping_randomized_with_rotation_and_random_bottleneck',
    'alchemy/all_fixed',
    'alchemy/all_fixed_w_shaping',
    'alchemy/evaluation_episodes/321',
)


class LoadFromDiskTest(test_utils.EnvironmentTestMixin, absltest.TestCase):

  def make_object_under_test(self):
    return dm_alchemy.load_from_disk(
        FLAGS.path,
        settings=dm_alchemy.EnvironmentSettings(
            seed=123, level_name=_TEST_LEVEL))


class AlchemyTest(parameterized.TestCase):

  @parameterized.parameters(*_TEST_LEVELS)
  def test_load_level(self, level_name):
    self.assertIsNotNone(
        dm_alchemy.load_from_disk(
            FLAGS.path,
            settings=dm_alchemy.EnvironmentSettings(
                seed=123, level_name=level_name)))


if __name__ == '__main__':
  absltest.main()
