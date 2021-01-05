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
"""Create graph from unity type."""

from typing import Sequence

from dm_alchemy.protos import alchemy_pb2
from dm_alchemy.types import graphs
from dm_alchemy.types import stones_and_potions


def create_graph(
    potions: Sequence[stones_and_potions.Potion],
    chemistry: alchemy_pb2.Chemistry
) -> graphs.Graph:
  """Creates the graph from the chemistry event and the existing potions.

  Args:
    potions: list of Potion objects
    chemistry: a Chemistry event containing bottleneck topology

  Returns:
    A Graph object
  """
  vertices = chemistry.stones
  potion_effects = chemistry.potions
  node_list = graphs.NodeList()
  edge_list = graphs.EdgeList()

  for vertex in vertices:
    idx = vertex.latent.index
    coord = vertex.latent.coordinates
    node_list.add_node(graphs.Node(idx, coord))

  for potion in potions:
    dimension = potion.dimension
    direction = potion.direction
    allowable_effects = [effect.reactions for effect in potion_effects if \
                         effect.label.dimension_index == dimension \
                         and effect.label.direction == direction][0]

    for effect in allowable_effects:
      from_node = node_list.get_node_by_idx(effect.from_stone_index)
      to_node = node_list.get_node_by_idx(effect.to_stone_index)
      edge_list.add_edge(from_node, to_node, potion)

  return graphs.Graph(node_list, edge_list)
