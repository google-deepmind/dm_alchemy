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

import itertools

from absl.testing import absltest
from absl.testing import parameterized
from dm_alchemy.ideal_observer import precomputed_maps
from dm_alchemy.types import stones_and_potions
import numpy as np


class StonesAndPotionsTest(parameterized.TestCase):

  perm_index_to_index = None
  index_to_perm_index = None

  @classmethod
  def setUpClass(cls):
    super(StonesAndPotionsTest, cls).setUpClass()
    # Load a precomputed map.
    cls.perm_index_to_index, cls.index_to_perm_index = (
        precomputed_maps.get_perm_index_conversion())

  @parameterized.parameters(
      (stones_and_potions.aligned_stone_from_index,
       stones_and_potions.AlignedStone.num_types),
      (stones_and_potions.perceived_potion_from_index,
       stones_and_potions.PerceivedPotion.num_types),
      (stones_and_potions.latent_stone_from_index,
       stones_and_potions.LatentStone.num_types),
      (stones_and_potions.latent_potion_from_index,
       stones_and_potions.LatentPotion.num_types),
      (stones_and_potions.stone_map_from_index,
       stones_and_potions.StoneMap.num_types),
      (stones_and_potions.potion_map_from_index,
       stones_and_potions.PotionMap.num_types,
       lambda x: {'index_to_perm_index': x.index_to_perm_index},
       lambda x: {'perm_index_to_index': x.perm_index_to_index}),
      (stones_and_potions.partial_stone_map_from_index,
       stones_and_potions.PartialStoneMap.num_types),
      (stones_and_potions.partial_potion_map_from_index, (
          stones_and_potions.PartialPotionMap.num_axis_assignments,
          stones_and_potions.PartialPotionMap.num_dir_assignments),
       lambda x: {'index_to_perm_index': x.index_to_perm_index},
       lambda x: {'perm_index_to_index': x.perm_index_to_index})
  )
  def test_index(
      self, from_index, num_indices, from_index_precomputed_args=None,
      to_index_precomputed_args=None):
    """Tests all valid indices converting to and from their type."""

    if from_index_precomputed_args is None:
      from_index_precomputed_args = lambda _: {}

    if to_index_precomputed_args is None:
      to_index_precomputed_args = lambda _: {}

    if isinstance(num_indices, tuple):
      unique_indices = itertools.product(*[range(i) for i in num_indices])
      expected_instances = np.prod(num_indices)
    else:
      unique_indices = range(num_indices)
      expected_instances = num_indices
    instances = set()
    for i in unique_indices:
      instance = from_index(i, **from_index_precomputed_args(self))
      instances.add(instance)
      back_to_index = instance.index(**to_index_precomputed_args(self))
      self.assertEqual(back_to_index, i)

    # All instances should be unique
    self.assertLen(instances, expected_instances)

if __name__ == '__main__':
  absltest.main()
