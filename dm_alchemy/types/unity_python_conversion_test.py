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
"""Tests for converting between unity and python representations."""

import functools
from typing import List

from absl.testing import absltest
from absl.testing import parameterized
from dm_alchemy.ideal_observer import precomputed_maps
from dm_alchemy.protos import alchemy_pb2
from dm_alchemy.types import graphs
from dm_alchemy.types import stones_and_potions
from dm_alchemy.types import unity_python_conversion
from dm_alchemy.types import utils
import numpy as np


def _make_tuple(i):
  return i,


def _unity_potions_given_constraint(
    constraint: graphs.Constraint) -> List[alchemy_pb2.PotionProperties]:
  """Return list of unity potion properties which encode the passed constraint."""
  graph = graphs.create_graph_from_constraint(constraint)
  # Use any potion map, it doesn't matter since we only care about reactions.
  pm = stones_and_potions.PotionMap(dim_map=[0, 1, 2], dir_map=[1, 1, 1])
  perceived_and_latent = [(pm.apply_inverse(l), l) for l in
                          stones_and_potions.possible_latent_potions()]
  return [unity_python_conversion.to_potion_unity_properties(p, l, graph)
          for p, l in perceived_and_latent]


def get_potion_tests():
  """Test cases for converting between potions and unity potion properties."""
  potion_tests = []
  for pm in stones_and_potions.possible_potion_maps(
      precomputed_maps.get_perm_index_conversion()[1]):
    potion_tests.append(
        ([(pm.apply_inverse(l), l) for l in
          stones_and_potions.possible_latent_potions()],
         functools.partial(
             unity_python_conversion.to_potion_unity_properties,
             # It shouldn't matter what graph we use for testing this part.
             graph=graphs.create_graph_from_constraint(
                 graphs.no_bottleneck_constraints()[0])),
         unity_python_conversion._potions_from_potion_unity_properties,
         lambda x: x, _make_tuple))
  return potion_tests


def from_stone_unity_properties(stone_properties, rotation):
  perceived_stone, _, latent_stone = (
      unity_python_conversion._from_stone_unity_properties(
          stone_properties, rotation))
  return perceived_stone, latent_stone


def get_stone_tests():
  """Test cases for converting between stones and unity stone properties."""
  stone_tests = []
  for rotation in stones_and_potions.possible_rotations():
    for sm in stones_and_potions.possible_stone_maps():
      stone_tests.append(
          ([(stones_and_potions.unalign(sm.apply_inverse(l), rotation), l)
            for l in stones_and_potions.possible_latent_stones()],
           unity_python_conversion.to_stone_unity_properties,
           functools.partial(from_stone_unity_properties, rotation=rotation),
           lambda x: x, _make_tuple))
  return stone_tests


def all_graphs():
  return [graphs.create_graph_from_constraint(g) for g in
          graphs.possible_constraints()]


def test_chemistries():
  """Return a subset of chemistries to test conversion."""
  chems = [
      utils.Chemistry(
          potion_map=stones_and_potions.all_fixed_potion_map(),
          stone_map=stones_and_potions.all_fixed_stone_map(),
          graph=graphs.create_graph_from_constraint(
              graphs.no_bottleneck_constraints()[0]),
          rotation=np.eye(3)),
      utils.Chemistry(
          potion_map=stones_and_potions.PotionMap([1, 0, 2], [1, 1, -1]),
          stone_map=stones_and_potions.StoneMap(np.array([-1, 1, -1])),
          graph=graphs.create_graph_from_constraint(
              graphs.bottleneck1_constraints()[0]),
          rotation=stones_and_potions.possible_rotations()[-1])]
  for r in stones_and_potions.possible_rotations():
    for sm in stones_and_potions.possible_stone_maps():
      chems.append(utils.Chemistry(
          potion_map=stones_and_potions.all_fixed_potion_map(),
          stone_map=sm,
          graph=graphs.create_graph_from_constraint(
              graphs.no_bottleneck_constraints()[0]),
          rotation=r))
  return chems


class UnityPythonConversionTest(parameterized.TestCase):

  @parameterized.parameters(
      [(stones_and_potions.possible_latent_stones(),
        unity_python_conversion.latent_stone_to_unity,
        unity_python_conversion._unity_to_latent_stone,
        _make_tuple, _make_tuple),
       (stones_and_potions.possible_latent_potions(),
        unity_python_conversion.latent_potion_to_unity,
        unity_python_conversion._unity_to_latent_potion,
        _make_tuple, _make_tuple),
       (stones_and_potions.possible_rotations(),
        unity_python_conversion.rotation_to_unity,
        unity_python_conversion.rotation_from_unity,
        _make_tuple, _make_tuple,
        lambda lhs, rhs: stones_and_potions.rotations_equal(*lhs, *rhs)),
       (test_chemistries(),
        unity_python_conversion.to_unity_chemistry,
        unity_python_conversion.from_unity_chemistry,
        _make_tuple, lambda x: x),
       # Test all graphs while keeping the potions constant
       # Compare constraint equality since graphs only compare equal if they use
       # the same node and edge objects.
       (all_graphs(),
        _unity_potions_given_constraint,
        unity_python_conversion.graphs_from_potion_unity_properties,
        lambda x: _make_tuple(graphs.constraint_from_graph(x)), _make_tuple)]
      # Test all potions while keeping the graph constant
      + get_potion_tests()
      # Test conversion of stones
      + get_stone_tests()
  )
  def test_back_and_forth(
      self, items, transform, inverse, post_process_i,
      post_process_transformed, equality=lambda lhs, rhs: lhs == rhs):
    """Test that transforming to and from unity types does not change anything.

    Since some of the transforms to test take or return multiple arguments we
    assume that all of them do. For functions which do not we use the post
    processing functions to make them tuples.

    Args:
      items: Set of items to transform.
      transform: Function which transforms them to unity type.
      inverse: Function which transforms unity type to python type.
      post_process_i: Function to post process the python type before passing it
        to the transform and before comparing it. This must return a tuple.
      post_process_transformed: Function to post process the unity type before
        passing it to the transform and before comparing it. This must return a
        tuple.
      equality: Function to check 2 python types are equal.
    """
    for i in items:
      # We assume these are all tuples after post processing so we can treat
      # them all the same.
      i = post_process_i(i)
      transformed = post_process_transformed(transform(*i))
      inverted = post_process_i(inverse(*transformed))
      self.assertTrue(equality(inverted, i))

if __name__ == '__main__':
  absltest.main()
