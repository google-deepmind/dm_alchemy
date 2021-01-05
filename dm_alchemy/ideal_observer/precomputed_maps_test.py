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
"""Tests for the precomputed maps used in the ideal observer."""

from absl.testing import absltest
from absl.testing import parameterized
from dm_alchemy.ideal_observer import precomputed_maps
from dm_alchemy.types import helpers
from dm_alchemy.types import stones_and_potions
import numpy as np

LatentStone = stones_and_potions.LatentStone
PerceivedPotion = stones_and_potions.PerceivedPotion
PotionMap = stones_and_potions.PotionMap
PartialPotionMap = stones_and_potions.PartialPotionMap
StoneMap = stones_and_potions.StoneMap
PartialStoneMap = stones_and_potions.PartialStoneMap
Stone = stones_and_potions.Stone
Potion = stones_and_potions.Potion


class PrecomputedMapsTest(parameterized.TestCase):

  no_effect_chem = None
  perm_index_to_index = None

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.perm_index_to_index, index_to_perm_index = (
        precomputed_maps.get_perm_index_conversion())
    cls.no_effect_chem = precomputed_maps.get_no_effect_from_partial_chem(
        index_to_perm_index)
    cls.partial_stone_map_to_stone_map = (
        precomputed_maps.get_partial_stone_map_to_stone_map())

  @parameterized.parameters(
      # If we know nothing about the potion map then we should not update any
      # entries in observed_no_effect_bits
      {'stone_map': stones_and_potions.all_fixed_stone_map(),
       'partial_potion_map': (
           stones_and_potions.no_knowledge_partial_potion_map()),
       'expected_observed_no_effect_bits': 0},
      # If we know the first dimension then observed no effect bits should have
      # a 1 for the potion which takes the stone out of the cube and 0s
      # elsewhere
      {'stone_map': stones_and_potions.all_fixed_stone_map(),
       'partial_potion_map': PartialPotionMap(
           [0, helpers.UNKNOWN, helpers.UNKNOWN],
           [1, helpers.UNKNOWN, helpers.UNKNOWN]),
       'potions': [PerceivedPotion(0, 1), PerceivedPotion(0, 1),
                   PerceivedPotion(1, 1), PerceivedPotion(2, -1),
                   PerceivedPotion(0, 1)],
       'stones': [LatentStone(np.array([1, -1, -1])),
                  LatentStone(np.array([1, -1, 1])),
                  LatentStone(np.array([1, -1, 1])),
                  LatentStone(np.array([1, -1, 1])),
                  LatentStone(np.array([-1, -1, 1]))],
       'observed_no_effect_vals': [1, 1, 0, 0]},
  )
  def test_no_effect_from_partial_chem(
      self, stone_map, partial_potion_map,
      expected_observed_no_effect_bits=None, potions=None, stones=None,
      observed_no_effect_vals=None):
    """Test that actions with no effect are correctly computed in a few cases."""

    partial_potion_map_index_0, partial_potion_map_index_1 = (
        partial_potion_map.index(self.perm_index_to_index))
    observed_no_effect_bits = self.no_effect_chem[
        stone_map.index(), partial_potion_map_index_0,
        partial_potion_map_index_1]
    if expected_observed_no_effect_bits is not None:
      self.assertEqual(
          observed_no_effect_bits, expected_observed_no_effect_bits)
    if potions is not None:
      for potion, stone, val in zip(potions, stones, observed_no_effect_vals):
        bit_num = stone.index() * PerceivedPotion.num_types + potion.index()
        bit_mask = 1 << bit_num
        expected_val = (observed_no_effect_bits & bit_mask) >> bit_num
        self.assertEqual(val, expected_val)

  @parameterized.parameters(
      # A fully specified partial stone map
      {'partial_stone_map': PartialStoneMap(np.array([-1, 1, -1])),
       'expected_stone_map_index': StoneMap(np.array([-1, 1, -1])).index()},
      # A partially specified partial stone map
      {'partial_stone_map': PartialStoneMap(np.array([-1, 1, helpers.UNKNOWN])),
       'expected_stone_map_index': -1},
  )
  def test_partial_stone_map_to_stone_map(
      self, partial_stone_map, expected_stone_map_index):

    stone_map_index = self.partial_stone_map_to_stone_map[
        partial_stone_map.index()]
    self.assertEqual(stone_map_index, expected_stone_map_index)

if __name__ == '__main__':
  absltest.main()
