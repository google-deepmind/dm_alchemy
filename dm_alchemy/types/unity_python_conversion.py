# Lint as python3
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
"""Code to convert between unity types and python types."""

import copy
import itertools
from typing import Any, Dict, Sequence, Tuple

from dm_alchemy.protos import alchemy_pb2
from dm_alchemy.protos import hypercube_pb2
from dm_alchemy.types import graphs
from dm_alchemy.types import stones_and_potions
from dm_alchemy.types import utils
import frozendict
import numpy as np

from dm_alchemy.protos import color_info_pb2
from dm_alchemy.protos import unity_types_pb2

PotionMap = stones_and_potions.PotionMap
StoneMap = stones_and_potions.StoneMap
AlignedStone = stones_and_potions.AlignedStone
PerceivedStone = stones_and_potions.PerceivedStone
PerceivedPotion = stones_and_potions.PerceivedPotion
LatentStone = stones_and_potions.LatentStone
LatentPotion = stones_and_potions.LatentPotion

MapsAndGraph = Tuple[PotionMap, StoneMap, graphs.Graph]

COLOR_TYPE = alchemy_pb2.PerceptualMappingApplicator.Type.COLOR
SIZE_TYPE = alchemy_pb2.PerceptualMappingApplicator.Type.SIZE
ROUNDNESS_TYPE = alchemy_pb2.PerceptualMappingApplicator.Type.ROUNDNESS

# Colours are defined in AlchemyColors.asset
_STONE_COLOURS = frozendict.frozendict({
    'purple': unity_types_pb2.Color(
        r=0.52156866, g=0.22745098, b=0.6313726, a=1.0),
    'blurple': unity_types_pb2.Color(
        r=0.2608, g=0.2667, b=0.5941, a=1.0),
    'blue': unity_types_pb2.Color(
        r=0.0, g=0.30588236, b=0.5568628, a=1.0)
})

_POTION_COLOURS = frozendict.frozendict({
    'green': unity_types_pb2.Color(
        r=0.24394463, g=0.6911765, b=0.35806578, a=1.0),
    'red': unity_types_pb2.Color(
        r=0.9647059, g=0.015686275, b=0.06666667, a=1.0),
    'yellow': unity_types_pb2.Color(
        r=0.9411765, g=0.84705883, b=0.078431375, a=1.0),
    'orange': unity_types_pb2.Color(
        r=0.9764706, g=0.4, b=0.10980392, a=1.0),
    'turquoise': unity_types_pb2.Color(
        r=0.21176471, g=0.72156864, b=0.7411765, a=1.0),
    'pink': unity_types_pb2.Color(
        r=0.9843137, g=0.43529412, b=0.43529412, a=1.0)
})

# This is the order of perceived axes in unity.
PERCEIVED_AXIS = (COLOR_TYPE, SIZE_TYPE, ROUNDNESS_TYPE)
AXIS_NUMBER = frozendict.frozendict({
    a: i for i, a in enumerate(PERCEIVED_AXIS)})

SIZE_NAME_AT_COORD = frozendict.frozendict(
    {-1: 'small', 0: 'medium', 1: 'large'})
_STONE_SIZES = frozendict.frozendict(
    {'small': 1.0, 'medium': 1.4, 'large': 1.8})
SIZE_AT_COORD = frozendict.frozendict(
    {coord: _STONE_SIZES[name] for coord, name in SIZE_NAME_AT_COORD.items()})
_COORD_AT_SIZE = frozendict.frozendict({v: k for k, v in SIZE_AT_COORD.items()})


ROUNDNESS_NAME_AT_COORD = frozendict.frozendict(
    {-1: 'pointy', 0: 'somewhat pointy', 1: 'round'})
_STONE_ROUNDNESSES = frozendict.frozendict(
    {'pointy': 0.0, 'somewhat pointy': 0.5, 'round': 1.0})
ROUNDNESS_AT_COORD = frozendict.frozendict(
    {coord: _STONE_ROUNDNESSES[name]
     for coord, name in ROUNDNESS_NAME_AT_COORD.items()})
_COORD_AT_ROUNDNESS = frozendict.frozendict({
    v: k for k, v in ROUNDNESS_AT_COORD.items()})


# The colour proto is not hashable so convert to a type which is.
def colour_proto_to_hashable(
    colour: unity_types_pb2.Color) -> Tuple[float, float, float, float]:
  return (round(colour.r, 2), round(colour.g, 2), round(colour.b, 2),
          round(colour.a, 2))

COLOUR_NAME_AT_COORD = frozendict.frozendict(
    {-1: 'purple', 0: 'blurple', 1: 'blue'})
COLOUR_AT_COORD = frozendict.frozendict({
    coord: _STONE_COLOURS[name]
    for coord, name in COLOUR_NAME_AT_COORD.items()})
_COORD_AT_COLOUR = frozendict.frozendict(
    {colour_proto_to_hashable(v): k for k, v in COLOUR_AT_COORD.items()})

POTION_COLOUR_AT_PERCEIVED_POTION = frozendict.frozendict({
    PerceivedPotion(0, 1): 'green',
    PerceivedPotion(0, -1): 'red',
    PerceivedPotion(1, 1): 'yellow',
    PerceivedPotion(1, -1): 'orange',
    PerceivedPotion(2, 1): 'turquoise',
    PerceivedPotion(2, -1): 'pink',
})
_PERCEIVED_POTION_AT_POTION_COLOUR = frozendict.frozendict({
    colour_proto_to_hashable(_POTION_COLOURS[v]): k
    for k, v in POTION_COLOUR_AT_PERCEIVED_POTION.items()})


def get_colour_info(
    name_and_colour: Tuple[str, unity_types_pb2.Color]
) -> color_info_pb2.ColorInfo:
  return color_info_pb2.ColorInfo(
      color=name_and_colour[1], name=name_and_colour[0])


def latent_stone_to_unity(
    latent_stone: LatentStone) -> hypercube_pb2.HypercubeVertex:
  return hypercube_pb2.HypercubeVertex(
      index=latent_stone.index(),
      coordinates=latent_stone.latent_coords.tolist())


def _unity_to_latent_stone(
    latent: hypercube_pb2.HypercubeVertex) -> LatentStone:
  # Use numpy object type to store python ints rather than numpy ints.
  return LatentStone(np.array([int(coord) for coord in latent.coordinates],
                              dtype=np.object))


def perceptual_features(perceived_stone: PerceivedStone) -> Dict[str, Any]:
  return {
      'size': SIZE_AT_COORD[perceived_stone.perceived_coords[AXIS_NUMBER[
          SIZE_TYPE]]],
      'roundness': ROUNDNESS_AT_COORD[perceived_stone.perceived_coords[
          AXIS_NUMBER[ROUNDNESS_TYPE]]],
      'color': COLOUR_AT_COORD[perceived_stone.perceived_coords[AXIS_NUMBER[
          COLOR_TYPE]]],
  }


def to_stone_unity_properties(
    perceived_stone: PerceivedStone, latent_stone: LatentStone
) -> alchemy_pb2.StoneProperties:
  """Convert a perceived and latent stone to StoneProperties."""

  return alchemy_pb2.StoneProperties(
      reward=15 if perceived_stone.reward > 2 else perceived_stone.reward,
      latent=latent_stone_to_unity(latent_stone),
      **perceptual_features(perceived_stone))


def unity_to_perceived_stone(
    stone_properties: alchemy_pb2.StoneProperties
) -> PerceivedStone:
  """Convert StoneProperties to a perceived stone."""
  size = _COORD_AT_SIZE[round(stone_properties.size, 1)]
  roundness = _COORD_AT_ROUNDNESS[round(stone_properties.roundness, 1)]
  colour = _COORD_AT_COLOUR[colour_proto_to_hashable(stone_properties.color)]
  # Use numpy object type to store python ints rather than numpy ints.
  perceived_coords = np.array([0, 0, 0], dtype=np.float)
  perceived_coords[AXIS_NUMBER[SIZE_TYPE]] = size
  perceived_coords[AXIS_NUMBER[ROUNDNESS_TYPE]] = roundness
  perceived_coords[AXIS_NUMBER[COLOR_TYPE]] = colour
  latent_stone = _unity_to_latent_stone(stone_properties.latent)
  return PerceivedStone(latent_stone.reward(), perceived_coords)


def _from_stone_unity_properties(
    stone_properties: alchemy_pb2.StoneProperties,
    rotation: np.ndarray
) -> Tuple[PerceivedStone, AlignedStone, LatentStone]:
  """Convert StoneProperties to a perceived and latent stone."""
  latent_stone = _unity_to_latent_stone(stone_properties.latent)
  perceived_stone = unity_to_perceived_stone(stone_properties)
  aligned_stone = stones_and_potions.align(perceived_stone, rotation)
  return perceived_stone, aligned_stone, latent_stone


def latent_potion_to_unity(
    latent_potion: LatentPotion) -> hypercube_pb2.EdgeLabel:
  if latent_potion.latent_dir == 1:
    direction = hypercube_pb2.EdgeLabel.Direction.POSITIVE
  else:
    direction = hypercube_pb2.EdgeLabel.Direction.NEGATIVE
  return hypercube_pb2.EdgeLabel(
      dimension_index=latent_potion.latent_dim, direction=direction)


def _unity_to_latent_potion(
    edge_label: hypercube_pb2.EdgeLabel) -> LatentPotion:
  if edge_label.direction == hypercube_pb2.EdgeLabel.Direction.POSITIVE:
    latent_dir = 1
  else:
    latent_dir = -1
  return LatentPotion(
      latent_dim=edge_label.dimension_index, latent_dir=latent_dir)


def to_potion_unity_properties(
    perceived_potion: PerceivedPotion, latent_potion: LatentPotion,
    graph: graphs.Graph
) -> alchemy_pb2.PotionProperties:
  """Convert a perceived and latent potion and graph to PotionProperties."""
  colour_name = POTION_COLOUR_AT_PERCEIVED_POTION[perceived_potion]
  colour = get_colour_info((colour_name, _POTION_COLOURS[colour_name]))
  reactions = set()
  for startnode, endnodes in graph.edge_list.edges.items():
    expected_end_coords = copy.deepcopy(startnode.coords)
    expected_end_coords[latent_potion.latent_dim] = (
        startnode.coords[latent_potion.latent_dim] + 2 *
        latent_potion.latent_dir)
    expected_end_node = graph.node_list.get_node_by_coords(
        expected_end_coords)
    if not expected_end_node:
      continue
    if expected_end_node in endnodes:
      reactions.add((startnode.idx, expected_end_node.idx))
  reactions = [alchemy_pb2.PotionReaction(from_stone_index=from_stone,
                                          to_stone_index=to_stone)
               for from_stone, to_stone in reactions]

  sorted_reactions = sorted(
      reactions, key=lambda reaction: reaction.from_stone_index)
  return alchemy_pb2.PotionProperties(
      label=latent_potion_to_unity(latent_potion), reward=0, color=colour,
      glow_color=colour, reactions=sorted_reactions)


def unity_to_perceived_potion(
    potion: alchemy_pb2.PotionProperties
) -> PerceivedPotion:
  return _PERCEIVED_POTION_AT_POTION_COLOUR[
      colour_proto_to_hashable(potion.color.color)]


def _potions_from_potion_unity_properties(
    potion: alchemy_pb2.PotionProperties
) -> Tuple[PerceivedPotion, LatentPotion]:
  """Convert the unity representation to a perceived and latent potion."""
  return (unity_to_perceived_potion(potion),
          _unity_to_latent_potion(potion.label))


def graphs_from_potion_unity_properties(
    potions: Sequence[alchemy_pb2.PotionProperties]) -> graphs.Graph:
  """Convert a sequence of PotionProperties to a Graph."""
  node_list = graphs.all_nodes_in_graph()
  edge_list = graphs.EdgeList()
  for i, potion in enumerate(potions):
    _, latent = _potions_from_potion_unity_properties(potion)
    utils_potion = stones_and_potions.Potion(
        i, latent.latent_dim, latent.latent_dir)
    for reaction in potion.reactions:
      edge_list.add_edge(
          node_list.get_node_by_idx(reaction.from_stone_index),
          node_list.get_node_by_idx(reaction.to_stone_index),
          utils_potion)
  return graphs.Graph(node_list, edge_list)


def to_unity_chemistry(
    chemistry: utils.Chemistry
) -> Tuple[alchemy_pb2.Chemistry, alchemy_pb2.RotationMapping]:
  """Convert from python types to unity Chemistry object."""
  # Latent stones and potions are always in the same places.
  latent_stones = stones_and_potions.possible_latent_stones()
  latent_potions = stones_and_potions.possible_latent_potions()

  # Apply the dimension swapping map between latent stones in unity and latent
  # stones in python (see from_unity_chemistry for more explanation).
  python_to_unity = PythonToUnityDimMap(chemistry)
  python_latent_stones = [python_to_unity.apply_to_stone(latent_stone)
                          for latent_stone in latent_stones]
  python_latent_potions = [python_to_unity.apply_to_potion(latent_potion)
                           for latent_potion in latent_potions]

  # Apply the stone map to them to get perceptual stones.
  aligned_stones = [chemistry.stone_map.apply_inverse(stone)
                    for stone in python_latent_stones]
  perceived_stones = [
      stones_and_potions.unalign(stone, chemistry.rotation)
      for stone in aligned_stones]
  unity_stones = [to_stone_unity_properties(perceived, latent)
                  for perceived, latent in zip(perceived_stones, latent_stones)]

  # Apply the potion map to them to get perceptual potions.
  perceived_potions = [chemistry.potion_map.apply_inverse(potion)
                       for potion in python_latent_potions]

  unity_potions = [
      to_potion_unity_properties(perceived, latent, python_to_unity.graph)
      for perceived, latent in zip(perceived_potions, latent_potions)]

  unity_chemistry = alchemy_pb2.Chemistry(
      stones=unity_stones, potions=unity_potions)
  rotation_mapping = rotation_to_unity(python_to_unity.rotation)

  return unity_chemistry, rotation_mapping


def rotation_from_unity(
    rotation_mapping: alchemy_pb2.RotationMapping
) -> np.ndarray:
  """Get the transformation to undo rotation from unity."""
  # Rotate back
  angles = [-int(rotation_mapping.rotation_angles.x),
            -int(rotation_mapping.rotation_angles.y),
            -int(rotation_mapping.rotation_angles.z)]
  return stones_and_potions.rotation_from_angles(angles)


def rotation_to_unity(rotation: np.ndarray) -> alchemy_pb2.RotationMapping:
  """Convert the transformation to undo rotation to unity."""
  angles = stones_and_potions.rotation_to_angles(rotation)
  return alchemy_pb2.RotationMapping(rotation_angles=unity_types_pb2.Vector3(
      **{axis: -round(a) for axis, a in zip('xyz', angles)}))


def potion_map_from_potions(
    latent_potions: Sequence[LatentPotion],
    perceived_potions: Sequence[PerceivedPotion]
) -> PotionMap:
  """Calculate potion map relating latent and perceived potions."""
  dimension_map = [-1, -1, -1]
  direction_map = [0, 0, 0]
  for perceived_potion, latent_potion in zip(perceived_potions, latent_potions):
    dimension_map[perceived_potion.perceived_dim] = latent_potion.latent_dim
    if latent_potion.latent_dir == perceived_potion.perceived_dir:
      direction_map[latent_potion.latent_dim] = 1
    else:
      direction_map[latent_potion.latent_dim] = -1
  return PotionMap(dim_map=dimension_map, dir_map=direction_map)


def _get_aligned_coords_matching_latent(
    python_stones: Sequence[Tuple[PerceivedStone, AlignedStone, LatentStone]],
    latent_coords: Sequence[int]
) -> np.ndarray:
  return [aligned_stone.aligned_coords.astype(np.int)
          for _, aligned_stone, latent_stone in python_stones
          if latent_stone.latent_coords.tolist() == latent_coords][0]


def find_dim_map_and_stone_map(
    chemistry: utils.Chemistry
) -> Tuple[np.ndarray, StoneMap, np.ndarray]:
  """Find a dimension map and stone map which map latent stones to perceived."""

  latent_stones = stones_and_potions.possible_latent_stones()
  aligned_stones = [chemistry.stone_map.apply_inverse(stone)
                    for stone in latent_stones]
  perceived_stones = [stones_and_potions.unalign(stone, chemistry.rotation)
                      for stone in aligned_stones]

  for dim_map in [np.eye(3, dtype=np.int)[p, :] for p in itertools.permutations(
      [0, 1, 2])]:
    for stone_map in stones_and_potions.possible_stone_maps():
      sm = np.diag(stone_map.latent_pos_dir.astype(np.int))
      # Since we do rotation before reflection in this case we must allow
      # rotation forwards and backwards to get all cases.
      # Because of the scaling this is not just the inverse matrix.
      inverse_rotation = stones_and_potions.rotation_from_angles(
          [-a for a in stones_and_potions.rotation_to_angles(
              chemistry.rotation)])
      for rotation in [chemistry.rotation, inverse_rotation]:
        all_match = True
        for ls, ps in zip(latent_stones, perceived_stones):
          new_ls = np.matmul(dim_map, ls.latent_coords.astype(np.int))
          ps_prime = np.matmul(sm, np.matmul(np.linalg.inv(rotation), new_ls))
          if not all(abs(a - b) < 0.0001 for a, b in zip(
              ps_prime, ps.perceived_coords.astype(np.int))):
            all_match = False
            break
        if all_match:
          return np.linalg.inv(dim_map), stone_map, rotation
  assert False, (
      'No dimension map and stone map takes latent stones to the passed '
      'perceived stones with the passed rotation.')


def _apply_dim_map_to_stone(
    dim_map: np.ndarray, latent_stone: LatentStone
) -> LatentStone:
  coords = np.rint(np.matmul(
      dim_map, latent_stone.latent_coords.astype(np.int)))
  return LatentStone(np.array([int(c) for c in coords], np.object))


def _apply_dim_map_to_potion(
    dim_map: np.ndarray, latent_potion: LatentPotion
) -> LatentPotion:
  return LatentPotion(
      np.where(dim_map[latent_potion.latent_dim, :])[0][0],
      latent_potion.latent_dir)


def _apply_dim_map_to_graph(
    dim_map: np.ndarray, graph: graphs.Graph
) -> graphs.Graph:
  """Swap latent dimensions in graph."""
  edge_list = graphs.EdgeList()
  for start_node, end_nodes in graph.edge_list.edges.items():
    start_coords = np.matmul(dim_map, np.array(start_node.coords)).tolist()
    new_start_node = graph.node_list.get_node_by_coords(start_coords)
    for end_node, edge in end_nodes.items():
      end_coords = np.matmul(dim_map, np.array(end_node.coords)).tolist()
      new_end_node = graph.node_list.get_node_by_coords(end_coords)
      new_potion = stones_and_potions.Potion(
          edge[1].idx, np.where(dim_map[edge[1].dimension, :])[0][0],
          edge[1].direction)
      edge_list.add_edge(new_start_node, new_end_node, new_potion)
  return graphs.Graph(graph.node_list, edge_list)


class PythonToUnityDimMap:
  """Convert from python method of mapping to unity method."""

  def __init__(self, chemistry: utils.Chemistry):
    self._chemistry = chemistry
    self._dim_map, self.stone_map, self.rotation = find_dim_map_and_stone_map(
        chemistry)
    self.graph = self._apply_to_graph(self._chemistry.graph)
    self.potion_map = self._apply_to_potion_map(self._chemistry.potion_map)

  def apply_to_stone(self, latent_stone: LatentStone) -> LatentStone:
    return _apply_dim_map_to_stone(self._dim_map, latent_stone)

  def apply_to_potion(self, latent_potion: LatentPotion) -> LatentPotion:
    return _apply_dim_map_to_potion(self._dim_map, latent_potion)

  def _apply_to_graph(self, graph: graphs.Graph) -> graphs.Graph:
    return _apply_dim_map_to_graph(self._dim_map, graph)

  def _apply_to_potion_map(self, potion_map: PotionMap) -> PotionMap:
    latent_potions = stones_and_potions.possible_latent_potions()
    new_latent_potions = [self.apply_to_potion(latent_potion)
                          for latent_potion in latent_potions]
    perceived_potions = [potion_map.apply_inverse(latent_potion)
                         for latent_potion in latent_potions]
    return potion_map_from_potions(new_latent_potions, perceived_potions)


def from_unity_chemistry(
    chemistry: alchemy_pb2.Chemistry,
    rotation_mapping: alchemy_pb2.RotationMapping
) -> utils.Chemistry:
  """Convert from unity Chemistry object to corresponding python types.

  Args:
    chemistry: A chemistry object received from the alchemy unity environment.
    rotation_mapping: A rotation mapping object received from the alchemy unity
      environment.

  Returns:
    A PotionMap describing the transformation from potion perceptual space to
      latent space.
    A StoneMap describing the transformation from stone aligned perceptual space
      to latent space.
    A Graph describing the available edges in latent space.
    A np.ndarray describing the rotation from stone aligned perceptual space to
      stone perceptual space.
  """

  # In unity the latent stones are (possibly) rotated and then "perceptual
  # mapping applicators" are applied to say how this is represented on screen,
  # e.g. -1 in the first latent dimension is purple and +1 is blue.
  # By only considering 7 possible rotations (0 rotation and 45 degrees
  # clockise or anticlockwise about each axis) and just considering in what
  # direction perceptual attributes change, when this is combined with the
  # mapping of potion pairs to latent space dimensions and assigning a direction
  # to that potion pair, we get all mappings which are 45 degrees offset on one
  # axis (note that latent variables have the same effect on the reward so
  # swapping latent space dimensions has no effect). We get duplicates because
  # after rotating, one dimension of the max reward stone will have value 0 so
  # reflecting about this does not change the value. However, the configuration
  # is such that the task distribution is as it would be if we avoided
  # duplicates.
  # An alternative way to generate all these mappings without the duplicates
  # would be to take the stones latent coordinates and first apply a mapping
  # which changes the positive direction and then rotate these positions by 45
  # degrees clockwise (excluding anticlockwise rotations).
  # It is easier to run algorithms like the ideal observer assuming the second
  # breakdown of the mapping because the rotation does not effect the best
  # action to take so we can take the perceived coordinates and undo the
  # rotation using any plausible rotation (even if it is not the correct one)
  # and then maintain a belief state over the remaining aspects of the
  # chemistry and update the belief state if we find the rotation was wrong.
  # We can switch between these equivalent breakdowns by possibly rotating in
  # the opposite direction.

  # From unity we get
  # perceived_stone = sm * r * latent_stone
  # where r rotates plus or minus 45 degrees and sm changes directions, we want
  # perceived_stone = r_prime * sm * latent_stone
  # where r_prime is rotating clockwise about the axis that r rotates around.
  rotation = rotation_from_unity(rotation_mapping)
  abs_rotation = stones_and_potions.rotation_from_angles(
      [-abs(a) for a in stones_and_potions.rotation_to_angles(rotation)])
  python_stones = [_from_stone_unity_properties(stone, abs_rotation)
                   for stone in chemistry.stones]
  python_potions = [_potions_from_potion_unity_properties(potion)
                    for potion in chemistry.potions]
  graph = graphs_from_potion_unity_properties(chemistry.potions)

  # So sm_prime is diagonal with elements in {-1, 1} and dim_map is such that
  # the sum of each row and each column is 1 with non zero elements 1.
  # Let a := sm_prime * dim_map
  # a := [a11 a12 a13]
  #      [a21 a22 a23]
  #      [a31 a32 a33]
  # a * [1, 1, 1] = [a11 + a12 + a13, a21 + a22 + a23, a31 + a32 + a33]
  sum_of_each_row = _get_aligned_coords_matching_latent(
      python_stones, [1, 1, 1])
  stone_map = StoneMap(pos_dir=sum_of_each_row)
  sm_prime = np.diag(sum_of_each_row)
  # a * [1, 1, 1] - a * [-1, 1, 1] = 2 * [a11, a21, a31]
  first_column = ((sum_of_each_row - _get_aligned_coords_matching_latent(
      python_stones, [-1, 1, 1]))/2).astype(np.int)
  second_column = ((sum_of_each_row - _get_aligned_coords_matching_latent(
      python_stones, [1, -1, 1]))/2).astype(np.int)
  third_column = ((sum_of_each_row - _get_aligned_coords_matching_latent(
      python_stones, [1, 1, -1]))/2).astype(np.int)
  a = np.hstack((first_column.reshape((3, 1)), second_column.reshape((3, 1)),
                 third_column.reshape((3, 1))))
  dim_map = np.rint(np.matmul(np.linalg.inv(sm_prime), a)).astype(np.int)

  latent_stones = [latent_stone for _, _, latent_stone in python_stones]
  aligned_stones = [aligned_stone for _, aligned_stone, _ in python_stones]
  latent_stones = [_apply_dim_map_to_stone(dim_map, latent_stone)
                   for latent_stone in latent_stones]
  latent_potions = [latent_potion for _, latent_potion in python_potions]
  latent_potions = [_apply_dim_map_to_potion(dim_map, latent_potion)
                    for latent_potion in latent_potions]
  perceived_potions = [perceived_potion
                       for perceived_potion, _ in python_potions]
  graph = _apply_dim_map_to_graph(dim_map, graph)

  for aligned_stone, latent_stone in zip(aligned_stones, latent_stones):
    assert stone_map.apply(aligned_stone) == latent_stone, (
        'Applying the stone map to the aligned stone did not give the '
        'expected latent stone.\n{aligned_stone}\n{latent_stone}\n'
        '{stone_map}\n{chemistry}'.format(
            aligned_stone=aligned_stone, latent_stone=latent_stone,
            stone_map=stone_map, chemistry=chemistry))

  potion_map = potion_map_from_potions(latent_potions, perceived_potions)
  for perceived_potion, latent_potion in zip(perceived_potions, latent_potions):
    assert potion_map.apply(perceived_potion) == latent_potion, (
        'Applying the potion map to the perceived potion did not give the '
        'expected latent potion.{perceived_potion}\n{latent_potion}\n'
        '{potion_map}\n{chemistry}'.format(
            perceived_potion=perceived_potion, latent_potion=latent_potion,
            potion_map=potion_map, chemistry=chemistry))

  return utils.Chemistry(potion_map, stone_map, graph, abs_rotation)
