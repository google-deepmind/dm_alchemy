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
"""Precompute variables mapping inputs to outputs for complex functions."""

import collections
import copy
import functools
import itertools
import os
from typing import Any, Mapping, Optional, Sequence, Tuple

from absl import logging
import dataclasses
from dm_alchemy import io
from dm_alchemy.encode import precomputed_maps_pb2
from dm_alchemy.encode import precomputed_maps_proto_conversion
from dm_alchemy.ideal_observer import helpers
from dm_alchemy.types import graphs
from dm_alchemy.types import helpers as types_helpers
from dm_alchemy.types import stones_and_potions
import frozendict
import numpy as np


@dataclasses.dataclass
class PrecomputedMaps:
  """Functions to get the observations for different content types for each element."""
  graphs_list: np.ndarray
  graph_index_distr: np.ndarray
  partial_graph_to_matching_graphs: np.ndarray
  partial_graph_update: np.ndarray
  stone_to_reward: np.ndarray
  drop_reward: np.ndarray
  partial_graph_index_to_possible_index: Mapping[int, int]
  graphs_with_edge: np.ndarray
  edge_exists: np.ndarray
  stone_maps: np.ndarray
  potion_maps: np.ndarray
  possible_latent_dims: np.ndarray
  poss_p_maps: np.ndarray
  poss_s_maps: np.ndarray
  react_result: np.ndarray
  possible_latent_dirs: np.ndarray
  partial_potion_map_update: np.ndarray
  partial_stone_map_update: np.ndarray
  potion_masks: np.ndarray
  potion_to_pair: np.ndarray
  perm_index_to_index: np.ndarray
  index_to_perm_index: np.ndarray
  missing_edge_no_change: np.ndarray
  update_partial_graph_no_change: np.ndarray
  partial_stone_map_to_stone_map: np.ndarray
  no_effect_from_partial_chem: np.ndarray

  def __deepcopy__(self, memo):
    # Don't deepcopy precomputed maps as it takes too long and uses too much
    # memory and the contents never change after construction so we only need 1.
    return copy.copy(self)

  def save(self, folder):
    """Saves the precomputed maps to serialized protos in the folder passed in."""
    precomputed_maps_proto_conversion.write_graph_array(
        self.graphs_list, folder, 'graphs_list')

    for int_array, name in [
        (self.stone_to_reward, 'stone_to_reward'),
        (self.drop_reward, 'drop_reward'),
        (self.edge_exists, 'edge_exists'),
        (self.stone_maps, 'stone_maps'),
        (self.potion_maps, 'potion_maps'),
        (self.react_result, 'react_result'),
        (self.partial_stone_map_update, 'partial_stone_map_update'),
        (self.potion_to_pair, 'potion_to_pair'),
        (self.perm_index_to_index, 'perm_index_to_index'),
        (self.index_to_perm_index, 'index_to_perm_index'),
        (self.missing_edge_no_change, 'missing_edge_no_change'),
        (self.update_partial_graph_no_change, 'update_partial_graph_no_change'),
        (self.partial_stone_map_to_stone_map, 'partial_stone_map_to_stone_map'),
    ]:
      precomputed_maps_proto_conversion.write_int_array(int_array, folder, name)

    for int_array, name in [
        (self.partial_graph_to_matching_graphs,
         'partial_graph_to_matching_graphs'),
        (self.graphs_with_edge, 'graphs_with_edge'),
        (self.potion_masks, 'potion_masks'),
        (self.no_effect_from_partial_chem, 'no_effect_from_partial_chem'),
    ]:
      precomputed_maps_proto_conversion.write_bitfield_array(
          int_array, folder, name)

    precomputed_maps_proto_conversion.write_float_array(
        self.graph_index_distr, folder, 'graph_index_distr')

    for int_array, name in [
        (self.possible_latent_dims, 'possible_latent_dims'),
        (self.partial_graph_update, 'partial_graph_update'),
        (self.poss_p_maps, 'poss_p_maps'),
        (self.poss_s_maps, 'poss_s_maps'),
    ]:
      precomputed_maps_proto_conversion.write_list_ints_array(
          int_array, folder, name)

    precomputed_maps_proto_conversion.write_possible_latent_dirs(
        self.possible_latent_dirs, folder, 'possible_latent_dirs')
    precomputed_maps_proto_conversion.write_partial_potion_map_update(
        self.partial_potion_map_update, folder, 'partial_potion_map_update')

    proto = precomputed_maps_pb2.PartialGraphIndexToPossibleIndex(
        entries=self.partial_graph_index_to_possible_index)
    io.write_proto(
        os.path.join(folder, 'partial_graph_index_to_possible_index'),
        proto.SerializeToString())


def _load_from_folder(folder):
  """Loads precomputed maps from serialized protos in the folder passed in."""
  kwargs = {'graphs_list': precomputed_maps_proto_conversion.load_graph_array(
      folder, 'graphs_list')}

  for name in [
      'stone_to_reward', 'drop_reward', 'edge_exists', 'stone_maps',
      'potion_maps', 'react_result', 'partial_stone_map_update',
      'potion_to_pair', 'perm_index_to_index', 'index_to_perm_index',
      'missing_edge_no_change', 'update_partial_graph_no_change',
      'partial_stone_map_to_stone_map']:
    kwargs[name] = precomputed_maps_proto_conversion.load_int_array(
        folder, name)

  for name in [
      'partial_graph_to_matching_graphs', 'graphs_with_edge', 'potion_masks',
      'no_effect_from_partial_chem']:
    kwargs[name] = precomputed_maps_proto_conversion.load_bitfield_array(
        folder, name)

  for name in [
      'possible_latent_dims', 'poss_p_maps', 'poss_s_maps',
      'partial_graph_update']:
    kwargs[name] = precomputed_maps_proto_conversion.load_list_ints_array(
        folder, name)

  kwargs['graph_index_distr'] = (
      precomputed_maps_proto_conversion.load_float_array(
          folder, 'graph_index_distr'))
  kwargs['possible_latent_dirs'] = (
      precomputed_maps_proto_conversion.load_possible_latent_dirs(
          folder, 'possible_latent_dirs'))
  kwargs['partial_potion_map_update'] = (
      precomputed_maps_proto_conversion.load_partial_potion_map_update(
          folder, 'partial_potion_map_update'))
  serialized = io.read_proto(os.path.join(
      folder, 'partial_graph_index_to_possible_index'))
  proto = precomputed_maps_pb2.PartialGraphIndexToPossibleIndex.FromString(
      serialized)
  kwargs['partial_graph_index_to_possible_index'] = proto.entries
  return PrecomputedMaps(**kwargs)


# Alias these for readability
AlignedStone = stones_and_potions.AlignedStone
AlignedStoneIndex = stones_and_potions.AlignedStoneIndex
PerceivedPotion = stones_and_potions.PerceivedPotion
PerceivedPotionIndex = stones_and_potions.PerceivedPotionIndex
LatentStone = stones_and_potions.LatentStone
LatentPotion = stones_and_potions.LatentPotion
StoneMap = stones_and_potions.StoneMap
PotionMap = stones_and_potions.PotionMap
PartialStoneMap = stones_and_potions.PartialStoneMap
PartialPotionMap = stones_and_potions.PartialPotionMap
PartialGraph = graphs.PartialGraph

aligned_stone_from_index = stones_and_potions.aligned_stone_from_index
perceived_potion_from_index = stones_and_potions.perceived_potion_from_index
latent_stone_from_index = stones_and_potions.latent_stone_from_index
latent_potion_from_index = stones_and_potions.latent_potion_from_index
stone_map_from_index = stones_and_potions.stone_map_from_index
potion_map_from_index = stones_and_potions.potion_map_from_index
partial_stone_map_from_index = stones_and_potions.partial_stone_map_from_index
partial_potion_map_from_index = stones_and_potions.partial_potion_map_from_index
partial_graph_from_index = graphs.partial_graph_from_index


_SIMPLE_TYPE_COUNT = frozendict.frozendict({
    'PotionMap': PotionMap.num_types,
    'StoneMap': StoneMap.num_types,
    'LatentPotion': LatentPotion.num_types,
    'LatentStone': LatentStone.num_types,
    'PerceivedPotion': PerceivedPotion.num_types,
    'AlignedStone': AlignedStone.num_types,
    'PartialPotionMap_dim': PartialPotionMap.num_axis_assignments,
    'PartialPotionMap_dir': PartialPotionMap.num_dir_assignments,
    'PartialStoneMap': PartialStoneMap.num_types,
    'dim': stones_and_potions.get_num_axes(),
    'dir': stones_and_potions.get_num_dirs(),
})


_SIMPLE_TYPE_RECONSTRUCTOR = frozendict.frozendict({
    'StoneMap': stone_map_from_index,
    'LatentPotion': latent_potion_from_index,
    'LatentStone': latent_stone_from_index,
    'PerceivedPotion': perceived_potion_from_index,
    'AlignedStone': aligned_stone_from_index,
    'PartialStoneMap': partial_stone_map_from_index,
    'dim': lambda x: x,
    'dir': stones_and_potions.index_to_dir,
})

# Dict of reconstructors for indices which are passed in additional data.
_INDICES_PASSED_IN_RECONSTRUCTOR = frozendict.frozendict({
    'graph_important_edges':
        lambda x: (latent_stone_from_index(x[0]), latent_stone_from_index(x[1]))
    ,
    'possible_partial_graph_indices': partial_graph_from_index,
    'nodes': lambda x: x
})


def partial_potion_map_part_index(data, dim=0, direction=0):
  return partial_potion_map_from_index(
      (dim, direction), data['index_to_perm_index'])


_RECONSTRUCTORS_REQUIRING_DATA = frozendict.frozendict({
    'PotionMap':
        lambda i, data: potion_map_from_index(i, data['index_to_perm_index']),
    'PartialPotionMap_dim':
        lambda i, data: partial_potion_map_part_index(data, dim=i),
    'PartialPotionMap_dir':
        lambda i, data: partial_potion_map_part_index(data, direction=i),
})


_TYPE_FROM_TUPLE_INDEX = frozendict.frozendict({
    'PartialPotionMap': (
        ('PartialPotionMap_dim', 'PartialPotionMap_dir'),
        lambda i, data: partial_potion_map_part_index(data, i[0], i[1])),
})

PRECOMPUTED_LEVEL_FILES_DIR = 'ideal_observer/data'


def _get_type_count(current_type: str, additional_data: Mapping[str, Any]):
  if current_type in _SIMPLE_TYPE_COUNT:
    return _SIMPLE_TYPE_COUNT[current_type]
  return len(additional_data[current_type])


def _get_indices_and_reconstructor(current_type, additional_data):
  """For a given type gets valid indices and a method to reconstruct from index."""
  if 'enumerated_' in current_type:
    index_gen, reconstructor = _get_indices_and_reconstructor(
        current_type.replace('enumerated_', ''), additional_data)
    return enumerate(index_gen), lambda x: reconstructor(x[1])
  if current_type in _SIMPLE_TYPE_RECONSTRUCTOR:
    return (range(_SIMPLE_TYPE_COUNT[current_type]),
            _SIMPLE_TYPE_RECONSTRUCTOR[current_type])
  if current_type in _INDICES_PASSED_IN_RECONSTRUCTOR:
    return (additional_data[current_type],
            _INDICES_PASSED_IN_RECONSTRUCTOR[current_type])
  if current_type in _RECONSTRUCTORS_REQUIRING_DATA:
    return (range(_SIMPLE_TYPE_COUNT[current_type]),
            functools.partial(_RECONSTRUCTORS_REQUIRING_DATA[current_type],
                              data=additional_data))
  if current_type in _TYPE_FROM_TUPLE_INDEX:
    sub_types, reconstructor = _TYPE_FROM_TUPLE_INDEX[current_type]
    sub_indices = []
    for sub_type in sub_types:
      index_gen, _ = _get_indices_and_reconstructor(sub_type, additional_data)
      sub_indices.append(index_gen)
    return itertools.product(*sub_indices), functools.partial(
        reconstructor, data=additional_data)


def _reconstructed_elements(
    to_map: Mapping[str, str],
    additional_data: Mapping[str, np.ndarray]):
  """Generator for map from indices to elements."""
  # Get one of the types in to_map and loop through all possibilities for it
  # recursively calling for the remaining entries.
  indices_and_reconstructors = [
      _get_indices_and_reconstructor(current_type, additional_data)
      for current_type in to_map.values()]
  names = to_map.keys()
  indices = [elt[0] for elt in indices_and_reconstructors]
  reconstructors = [elt[1] for elt in indices_and_reconstructors]
  reconstructed = []
  # Indices may be generators and we iterate through twice so we must make a
  # copy
  for type_indices, reconstructor in zip(
      copy.deepcopy(indices), reconstructors):
    reconstructed.append([reconstructor(i) for i in type_indices])
  for current_index, current_element in zip(
      itertools.product(*indices), itertools.product(*reconstructed)):
    # We have to make a copy of the element before returning it because if it is
    # mutable and gets changed we don't want the change to be there for later
    # iterations.
    yield (collections.OrderedDict(
        [(name, i) for name, i in zip(names, current_index)]),
           collections.OrderedDict(
               [(name, copy.deepcopy(e)) for name, e in zip(
                   names, current_element)]))


_RESULT_TYPE_TO_EMPTY_RESULT = {
    # Use numpy object type to store python ints rather than numpy ints.
    'int': lambda s: np.zeros(s, dtype=object),
    # Create an array with an empty list at each entry.
    'list': lambda s: np.frompyfunc(list, 0, 1)(np.empty(s, dtype=object)),
    'tuple': lambda s: np.frompyfunc(tuple, 0, 1)(np.empty(s, dtype=object)),
}


def _empty_result(to_map, result_type, additional_data=None):
  shape = []
  for current_type in to_map:
    if current_type in _TYPE_FROM_TUPLE_INDEX:
      shape.extend([_get_type_count(sub_type, additional_data) for sub_type in
                    _TYPE_FROM_TUPLE_INDEX[current_type][0]])
    else:
      shape.append(_get_type_count(current_type, additional_data))
  shape = tuple(shape)
  return _RESULT_TYPE_TO_EMPTY_RESULT[result_type](shape)


LoopHelper = collections.namedtuple('LoopHelper', 'empty_result gen')


def _precompute_loop_helper(
    to_map, result_type, additional_data=None, result_to_map=None):
  """Creates an empty results array and generator for indices and elements.

  Args:
    to_map: A list of types to map optionally with a name for the index and the
      element associated with each type. If no name is provided the type name
      itself will be used as the name. See functions below for example usages.
    result_type: The type of each element in the result matrix.
    additional_data: Additional data required to loop over the types passed in
      and reconstruct the elements.
    result_to_map: A list of types which index the result. If none is provided
      then it is assumed to be the same as to_map.

  Returns:
    A LoopHelper type containing an empty numpy array and a generator which will
    loop through all of the valid indices and elements.
  """
  # Passing a name is optional - if no name is passed then use the type string.
  to_map = collections.OrderedDict(
      [elt if isinstance(elt, tuple) else (elt, elt) for elt in to_map])
  if result_to_map is None:
    result_to_map = to_map.values()
  # Remove enumerated from result_to_map
  result_to_map = [elt.replace('enumerated_', '') for elt in result_to_map]
  empty_result = _empty_result(result_to_map, result_type, additional_data)
  gen = functools.partial(_reconstructed_elements, to_map, additional_data)
  return LoopHelper(empty_result, gen)


def get_partial_graph_update(
    all_graphs, graph_important_edges,
    possible_partial_graph_indices, partial_graph_index_to_possible_index
) -> np.ndarray:
  """Updates partial graph after seeing that edge exists."""
  # Create an array to hold results with an empty list at each entry.
  result, gen = _precompute_loop_helper(
      ['graph_important_edges', 'possible_partial_graph_indices'], 'list',
      additional_data={
          'graph_important_edges': graph_important_edges,
          'possible_partial_graph_indices': possible_partial_graph_indices},
      result_to_map=['LatentStone', 'LatentStone'])
  for indices, elements in gen():
    latent_stone_index, latent_result_index = indices['graph_important_edges']
    latent_stone, latent_result = elements['graph_important_edges']
    partial_graph = elements['possible_partial_graph_indices']
    partial_graph.add_edge(latent_stone, latent_result, graphs.KNOWN_EDGE)
    partial_graph.update(all_graphs)
    poss_index = partial_graph_index_to_possible_index[partial_graph.index()]
    result[latent_stone_index, latent_result_index].append(poss_index)
    result[latent_result_index, latent_stone_index].append(poss_index)

  return result


def get_partial_graph_to_matching_graphs(
    all_graphs, possible_partial_graph_indices: np.ndarray) -> np.ndarray:
  """Gets list of graphs matching the partial graph."""
  result, gen = _precompute_loop_helper(
      ['enumerated_possible_partial_graph_indices'], 'int',
      additional_data={
          'possible_partial_graph_indices': possible_partial_graph_indices})
  for indices, elements in gen():
    i, _ = indices['enumerated_possible_partial_graph_indices']
    partial_graph = elements['enumerated_possible_partial_graph_indices']
    matches = partial_graph.matching_graphs(all_graphs, return_indices=True)
    result[i] = helpers.list_to_bitfield(matches)
  return result


def get_graphs_with_edge(valid_graphs, index_to_perm_index) -> np.ndarray:
  """Array of bitfields of graphs which have the given edge given the maps."""
  nodes = graphs.all_nodes_in_graph()
  result, gen = _precompute_loop_helper(
      ['StoneMap', 'PotionMap', 'AlignedStone', 'PerceivedPotion'], 'int',
      additional_data={'index_to_perm_index': index_to_perm_index})
  for indices, elements in gen():
    stone_map = elements['StoneMap']
    potion_map = elements['PotionMap']
    aligned_stone = elements['AlignedStone']
    perceived_potion = elements['PerceivedPotion']
    latent_potion = potion_map.apply(perceived_potion)
    potion_in_stone_space = stone_map.apply_to_potion(latent_potion)
    start_node = nodes.get_node_by_coords(list(aligned_stone.aligned_coords))
    end_node_coords = copy.deepcopy(aligned_stone.aligned_coords)
    end_node_coords[potion_in_stone_space.latent_dim] += (
        2 * potion_in_stone_space.latent_dir)
    end_node_coord = end_node_coords[potion_in_stone_space.latent_dim]
    if end_node_coord < -1 or end_node_coord > 1:
      # Not in any graph
      result[tuple(indices.values())] = 0
      continue
    end_node = nodes.get_node_by_coords(list(end_node_coords))
    poss_graphs = [i for i, g in enumerate(valid_graphs)
                   if g.edge_list.has_edge(start_node, end_node)]
    graphs_bitfield = helpers.list_to_bitfield(poss_graphs)
    result[tuple(indices.values())] = graphs_bitfield
  return result


def get_edge_exists(possible_partial_graph_indices: np.ndarray) -> np.ndarray:
  """Checks if an edge exists given partial graph info."""
  graph_nodes = graphs.all_nodes_in_graph()
  result, gen = _precompute_loop_helper(
      ['enumerated_possible_partial_graph_indices', 'enumerated_nodes', 'dim'],
      'int', additional_data={
          'possible_partial_graph_indices': possible_partial_graph_indices,
          'nodes': graph_nodes.nodes})
  for indices, elements in gen():
    i, _ = indices['enumerated_possible_partial_graph_indices']
    partial_graph = elements['enumerated_possible_partial_graph_indices']
    start_node_index, start_node = indices['enumerated_nodes']
    dim = indices['dim']
    start_coords = start_node.coords
    end_coords = copy.deepcopy(start_coords)
    end_coords[dim] = -start_coords[dim]
    end_node_ = graph_nodes.get_node_by_coords(end_coords)
    assert end_node_ is not None
    end_node: graphs.Node = end_node_
    result[i, start_node_index, dim] = partial_graph.known_adj_mat[
        start_node_index, end_node.idx]
  return result


def get_possible_partial_graph_indices(
    graph_important_edges: Sequence[Tuple[int, int]],
    graphs_list: Sequence[graphs.Graph]
) -> np.ndarray:
  """Calculates an exhaustive list of possible partial graphs.

  This is smaller than the list of partial graphs we can represent because some
  partial graphs are impossible. For example graphs which are known to be
  disconnected.

  It is important to use only the possible partial graphs because this makes it
  practical to store maps over all possibilities.

  Args:
    graph_important_edges: List of the edges which may exist in a graph.
    graphs_list: List of all valid graphs.

  Returns:
    The list of partial graph indices.
  """
  def remaining_edges(g):
    ret = []
    for edge in graph_important_edges:
      if g.known_adj_mat[edge] == types_helpers.UNKNOWN:
        ret.append(edge)
    return ret

  # TODO(b/173785715): Start with what we can deduce from the graphs_list.
  to_expand = [PartialGraph().index()]
  visited = {PartialGraph().index()}

  while to_expand:
    current_node = to_expand[0]
    to_expand = to_expand[1:]
    current_graph = partial_graph_from_index(current_node)
    for e in remaining_edges(current_graph):
      for val in [0, 1]:
        new_partial = copy.deepcopy(current_graph)
        new_partial.known_adj_mat[e] = val
        new_partial.known_adj_mat[e[1], e[0]] = val
        new_partial.update(graphs_list)
        new_partial_index = new_partial.index()
        if new_partial_index not in visited:
          visited.add(new_partial_index)
          to_expand.append(new_partial_index)

  return np.array(list(sorted(visited)), dtype=object)


def get_poss_potion_maps_and_stone_maps(
    perm_index_to_index: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
  """Gets a list of potion and stone maps possible given an observation."""
  poss_stone_maps, gen = _precompute_loop_helper(
      [('stone', 'AlignedStone'), ('potion', 'PerceivedPotion'),
       ('result', 'AlignedStone')], 'list')
  # In this function we get 2 results at the same time so make another.
  poss_potion_maps = _empty_result(
      ['AlignedStone', 'PerceivedPotion', 'AlignedStone'], 'list')
  for indices, elements in gen():
    potion_maps, stone_maps = stones_and_potions.one_action_outcome(
        elements['stone'], elements['potion'], elements['result'],
        perm_index_to_index)
    poss_potion_maps[tuple(indices.values())] = potion_maps
    poss_stone_maps[tuple(indices.values())] = stone_maps
  return poss_potion_maps, poss_stone_maps


def get_possible_latent_dims(
    index_to_perm_index: np.ndarray
) -> np.ndarray:
  """Gets a list of possible latent dimensions given a potion and partial map."""
  result, gen = _precompute_loop_helper(
      ['PerceivedPotion', 'PartialPotionMap_dim'], 'list',
      additional_data={'index_to_perm_index': index_to_perm_index})
  for indices, elements in gen():
    partial_potion_map = elements['PartialPotionMap_dim']
    perceived_potion = elements['PerceivedPotion']
    result[tuple(indices.values())] = (
        partial_potion_map.possible_latent_dims(perceived_potion))
  return result


def get_react_result(
    possible_partial_graph_indices: np.ndarray, edge_exists: np.ndarray,
    drop_reward: np.ndarray
) -> np.ndarray:
  """Gets the resulting stone when applying a potion to a stone."""
  result, gen = _precompute_loop_helper(
      ['AlignedStone', 'dim', 'dir',
       'enumerated_possible_partial_graph_indices'], 'int',
      additional_data={
          'possible_partial_graph_indices': possible_partial_graph_indices})
  for indices, elements in gen():
    aligned_stone = elements['AlignedStone']
    latent_dim = elements['dim']
    latent_dir = elements['dir']
    aligned_stone_index = indices['AlignedStone']
    latent_dir_index = indices['dir']
    partial_graph_index, _ = indices[
        'enumerated_possible_partial_graph_indices']
    # If we know the edge doesn't exist do not consider the possibility
    # that the stone changes.
    if edge_exists[
        partial_graph_index, drop_reward[aligned_stone_index],
        latent_dim] == graphs.NO_EDGE:
      result[aligned_stone_index, latent_dim, latent_dir_index,
             partial_graph_index] = helpers.IMPOSSIBLE
    else:
      result[aligned_stone_index, latent_dim, latent_dir_index,
             partial_graph_index] = stones_and_potions.react(
                 aligned_stone, latent_dim, latent_dir).index()
  return result


def get_possible_latent_dirs(index_to_perm_index: np.ndarray) -> np.ndarray:
  """Gets a list of possible latent dimensions given maps and stone and potion."""
  result, gen = _precompute_loop_helper(
      ['PartialPotionMap_dir', 'PartialStoneMap', 'dim', 'PerceivedPotion',
       'AlignedStone'], 'tuple',
      additional_data={'index_to_perm_index': index_to_perm_index})
  for indices, elements in gen():
    partial_potion_map = elements['PartialPotionMap_dir']
    partial_stone_map = elements['PartialStoneMap']
    latent_dim = elements['dim']
    perceived_potion = elements['PerceivedPotion']
    aligned_stone = elements['AlignedStone']
    latent_dirs_stone_dirs = (
        stones_and_potions.possible_latent_dirs_and_stone_dirs(
            perceived_potion, latent_dim, partial_potion_map,
            partial_stone_map))
    result[tuple(indices.values())] = (
        stones_and_potions.latent_dirs_on_stone(
            aligned_stone, latent_dim, partial_stone_map,
            latent_dirs_stone_dirs))
  return result


def get_partial_potion_map_update(
    index_to_perm_index: np.ndarray, perm_index_to_index: np.ndarray
) -> np.ndarray:
  """Updates a partial potion map given an observation."""
  result, gen = _precompute_loop_helper(
      [('stone', 'AlignedStone'), ('potion', 'PerceivedPotion'),
       ('result', 'AlignedStone'), 'PartialPotionMap'], 'tuple',
      additional_data={'index_to_perm_index': index_to_perm_index})
  for indices, elements in gen():
    stone_index = indices['stone']
    potion_index = indices['potion']
    result_index = indices['result']
    partial_potion_map_index = indices['PartialPotionMap']
    stone = elements['stone']
    potion = elements['potion']
    result_stone = elements['result']
    partial_potion_map = elements['PartialPotionMap']
    result[stone_index, potion_index, result_index, partial_potion_map_index[0],
           partial_potion_map_index[1]] = (
               stones_and_potions.update_partial_potion_map(
                   stone, potion, result_stone, partial_potion_map,
                   perm_index_to_index))
  return result


def get_partial_stone_map_update() -> np.ndarray:
  """Updates a partial stone map given an observation."""
  result, gen = _precompute_loop_helper(
      [('stone', 'AlignedStone'), ('result', 'AlignedStone'),
       'PartialStoneMap'], 'int')
  for indices, elements in gen():
    partial_stone_map = elements['PartialStoneMap']
    result[tuple(indices.values())] = (
        stones_and_potions.update_partial_stone_map(
            elements['stone'], elements['result'], partial_stone_map))
  return result


def get_missing_edge_no_change(
    index_to_perm_index: np.ndarray,
    graph_important_edges: Sequence[Tuple[int, int]]
) -> np.ndarray:
  """Gets which edge is missing given a potion has no effect."""
  result, gen = _precompute_loop_helper(
      ['PartialStoneMap', 'PartialPotionMap', 'PerceivedPotion', 'LatentStone'],
      'int', additional_data={'index_to_perm_index': index_to_perm_index})
  for indices, elements in gen():
    partial_potion_map = elements['PartialPotionMap']
    partial_stone_map = elements['PartialStoneMap']
    potion = elements['PerceivedPotion']
    aligned_stone_coords = elements['LatentStone']
    partial_stone_map_index = indices['PartialStoneMap']
    partial_potion_map_index_0, partial_potion_map_index_1 = indices[
        'PartialPotionMap']
    potion_index = indices['PerceivedPotion']
    stone_index = indices['LatentStone']
    # If we can't map the potion into latent space we cannot tell which
    # edge is missing.
    if not partial_potion_map.can_map(potion):
      result[partial_stone_map_index, partial_potion_map_index_0,
             partial_potion_map_index_1, potion_index, stone_index] = -1
      continue
    # If we can't map the potion from latent space into stone perceptual
    # space we cannot tell which edge is missing.
    latent_potion = partial_potion_map.apply(potion)
    if partial_stone_map.latent_pos_dir[
        latent_potion.latent_dim] == types_helpers.UNKNOWN:
      result[partial_stone_map_index, partial_potion_map_index_0,
             partial_potion_map_index_1, potion_index, stone_index] = -1
      continue
    stone_space_potion = partial_stone_map.apply_to_potion(latent_potion)
    # If the stone value on the dimension that the potion should change
    # is the opposite of the potion direction then the stone should have
    # changed and therefore we can eliminate graphs containing the edge.
    if aligned_stone_coords.latent_coords[
        stone_space_potion.latent_dim] == stone_space_potion.latent_dir:
      result[partial_stone_map_index, partial_potion_map_index_0,
             partial_potion_map_index_1, potion_index, stone_index] = -1
      continue
    # Set the result to be the index of the edge which shouldn't be
    # there.
    expected_end_coords = copy.deepcopy(aligned_stone_coords.latent_coords)
    expected_end_coords[stone_space_potion.latent_dim] = -expected_end_coords[
        stone_space_potion.latent_dim]
    expected_end_index = stones_and_potions.LatentStone(
        expected_end_coords).index()
    missing_edge = -1
    edge_start_end = sorted((stone_index, expected_end_index))
    for edge_index, (i, j) in enumerate(graph_important_edges):
      if sorted((i, j)) == edge_start_end:
        missing_edge = edge_index
    assert missing_edge != -1, 'Missing edge doesn\'t exist'
    result[partial_stone_map_index, partial_potion_map_index_0,
           partial_potion_map_index_1, potion_index, stone_index] = missing_edge

  return result


def get_partial_stone_map_to_stone_map() -> np.ndarray:
  """If a partial stone map is fully known returns stone map otherwise -1."""
  result, gen = _precompute_loop_helper(['PartialStoneMap'], 'int')
  for indices, elements in gen():
    index = indices['PartialStoneMap']
    partial_stone_map = elements['PartialStoneMap']
    stone_maps = partial_stone_map.fill_gaps()
    if len(stone_maps) != 1:
      result[index] = -1
    else:
      result[index] = stone_maps[0].index()
  return result


def get_no_effect_from_partial_chem(
    index_to_perm_index: np.ndarray
) -> np.ndarray:
  """Gets bit mask for potions known to take a stone out of the latent cube."""
  result, gen = _precompute_loop_helper(
      ['StoneMap', 'PartialPotionMap'], 'int', additional_data={
          'index_to_perm_index': index_to_perm_index})
  for indices, elements in gen():
    stone_map = elements['StoneMap']
    stone_map_index = indices['StoneMap']
    partial_potion_map = elements['PartialPotionMap']
    partial_potion_map_index_0, partial_potion_map_index_1 = indices[
        'PartialPotionMap']
    # Go through perceived potion and perceived stone (without reward) and
    # update if we know there will be no effect.
    _, no_effect_gen = _precompute_loop_helper(
        ['PerceivedPotion', 'LatentStone'], 'int')
    no_effect_result = 0
    for no_effect_indices, no_effect_elements in no_effect_gen():
      perceived_potion = no_effect_elements['PerceivedPotion']
      aligned_stone_wo_reward = no_effect_elements['LatentStone']

      latent_stone = stone_map.apply(AlignedStone(
          0, aligned_stone_wo_reward.latent_coords))

      # If we can map the perceived potion to latent space, do so and see if it
      # has an effect.
      if not partial_potion_map.can_map(perceived_potion):
        continue
      latent_potion = partial_potion_map.apply(perceived_potion)
      if latent_potion.latent_dir == latent_stone.latent_coords[
          latent_potion.latent_dim]:
        no_effect_result |= 1 << (
            (no_effect_indices['LatentStone'] * PerceivedPotion.num_types) +
            no_effect_indices['PerceivedPotion'])
    result[stone_map_index, partial_potion_map_index_0,
           partial_potion_map_index_1] = no_effect_result

  return result


def get_update_partial_graph_no_change(
    all_graphs, possible_partial_graph_indices: np.ndarray,
    partial_graph_index_to_possible_index: Mapping[int, int],
    graph_important_edges: Sequence[Tuple[int, int]]
) -> np.ndarray:
  """Given a missing edge updates the partial graph."""
  graph_nodes = graphs.all_nodes_in_graph()
  result, gen = _precompute_loop_helper(
      ['enumerated_possible_partial_graph_indices',
       'enumerated_graph_important_edges'], 'int',
      additional_data={
          'possible_partial_graph_indices': possible_partial_graph_indices,
          'graph_important_edges': graph_important_edges})
  for indices, elements in gen():
    edge_index, (start_node, end_node) = indices[
        'enumerated_graph_important_edges']
    poss_index, _ = indices[
        'enumerated_possible_partial_graph_indices']
    partial_graph = elements['enumerated_possible_partial_graph_indices']
    start_stone = stones_and_potions.LatentStone(np.array(
        graph_nodes.nodes[start_node].coords))
    end_stone = stones_and_potions.LatentStone(np.array(
        graph_nodes.nodes[end_node].coords))
    partial_graph.add_edge(start_stone, end_stone, graphs.NO_EDGE)
    partial_graph.update(all_graphs)
    result[poss_index, edge_index] = partial_graph_index_to_possible_index[
        partial_graph.index()]
  return result


def get_perm_index_conversion() -> Tuple[np.ndarray, np.ndarray]:
  """Gets maps to convert between different indices representing permutations.

  We make a map from an index computed by treating each entry in the permutation
  as being between 0 and len(perm) - 1 (of which there are
  len(perm) ^ len(perm)) to an index between 0 and len(perm)! - 1.
  len(perm) is 3 so this is not large.

  Returns:
    Map from index which treats entries as independent to compact index, the
    inverse.
  """
  num_axes = stones_and_potions.get_num_axes()
  # Use numpy object type to store python ints rather than numpy ints.
  perm_index_to_index = np.array([-1 for _ in range(num_axes ** num_axes)],
                                 dtype=object)
  for i, perm in enumerate(itertools.permutations(range(num_axes))):
    perm_index_to_index[np.ravel_multi_index(
        tuple(perm), tuple(num_axes for _ in range(num_axes)))] = i

  # Make the inverse map.
  index_to_perm_index = np.array(
      [int(np.ravel_multi_index(
          tuple(perm), tuple(num_axes for _ in range(num_axes))))
       for perm in itertools.permutations(range(3))], dtype=object)

  return perm_index_to_index, index_to_perm_index


def constraints_to_filename(
    constraints: Sequence[graphs.Constraint],
    poss_stone_maps: Sequence[stones_and_potions.StoneMap],
    poss_potion_maps: Sequence[stones_and_potions.PotionMap]
) -> str:
  """Converts a sequence of constraints and possible maps to a filename.

  This removes characters like * and - and ensures a list with the same number
  of constraints is the same length. Each constraint becomes 6 letters long with
  S (i.e. star) substituted for *, N (i.e. negative) substituted for -1 and P
  (i.e. positive) substituted for 1. Consecutive constraints are separated by a
  / and the sequence of constraints is lexicographically sorted to ensure that
  two sequences of constraints which differ only in order are represented by the
  same string.

  Stone and potion maps are converted to indices and the represented as a
  sequence of ranges, eg. 0-7/1,8,16-24,32.

  Args:
    constraints: A sequence of graphs.Constraints.
    poss_stone_maps: A sequence of possible stone maps.
    poss_potion_maps: A sequence of possible potion maps.

  Returns:
    A string with each constraint and possible stone map and potion map.
  """
  def constraint_to_str(constraint: graphs.Constraint) -> str:
    remapping = {'*': 'S', '-1': 'N', '1': 'P'}
    all_dims = []
    for i, dim in enumerate(constraint):
      constr = dim[:i] + dim[i + 1:]
      all_dims.append(''.join([remapping[c] for c in constr]))
    return ''.join(all_dims)

  def seq_ints_to_str(seq: Sequence[int]) -> str:
    """Convert a sequence of ints to a string."""
    ranges = []
    start_i, prev_i = None, None
    for i in seq:
      if start_i is None:
        start_i, prev_i = i, i
        continue
      if i != prev_i + 1:
        ranges.append((start_i, prev_i))
        start_i = i
      prev_i = i
    ranges.append((start_i, prev_i))
    return ','.join(str(s) + ('' if e == s else '-' + str(e)) for s, e in
                    ranges)

  perm_index_to_index, _ = get_perm_index_conversion()

  return ('/'.join(sorted(constraint_to_str(c) for c in constraints)) + '/' +
          seq_ints_to_str([s.index() for s in poss_stone_maps]) + '/' +
          seq_ints_to_str([p.index(perm_index_to_index) for p in
                           poss_potion_maps]))


def load_from_level_name(level_name: str) -> Optional[PrecomputedMaps]:
  """Loads precomputed for the level name passed if it exists."""
  # All levels are in alchemy and this is not included in the precomputed.pkl
  # file paths so remove this from the level name if it is included.
  if level_name.startswith('alchemy/'):
    level_name = level_name.replace('alchemy/', '')
  # Precomputed maps refer to the mapping between aligned stones and latent
  # stones so any rotation does not affect them so ignore it.
  # There are a few different ways of specifying rotation in the level name.
  level_name = level_name.replace('rotation_and_', '')
  level_name = level_name.replace('with_rotation', '')
  level_name = level_name.replace('fixed_with', 'fixed')
  level_name = level_name.replace('rotate_color_shape', '')
  level_name = level_name.replace('rotate_color_size', '')
  level_name = level_name.replace('rotate_size_shape', '')
  precomputed_folder = os.path.join(PRECOMPUTED_LEVEL_FILES_DIR, level_name)
  return _load_from_folder(precomputed_folder)


def get_precomputed_maps(
    constraints: Optional[Sequence[graphs.Constraint]] = None,
    poss_stone_maps: Optional[Sequence[stones_and_potions.StoneMap]] = None,
    poss_potion_maps: Optional[Sequence[stones_and_potions.PotionMap]] = None,
) -> PrecomputedMaps:
  """Precomputes a set of maps to make running the ideal observer faster."""

  # Constraints must be specified in stone perceptual space.
  if constraints is None:
    constraints = graphs.possible_constraints()

  if poss_stone_maps is None:
    poss_stone_maps = stones_and_potions.possible_stone_maps()

  perm_index_to_index, index_to_perm_index = get_perm_index_conversion()

  if poss_potion_maps is None:
    poss_potion_maps = stones_and_potions.possible_potion_maps(
        index_to_perm_index)

  logging.info('Computing precomputed maps.')

  # Everywhere below we use numpy object type to store python ints rather than
  # numpy ints so that we get arbitrary precision which allows us to make
  # bitfields easily.

  stone_maps = np.array([s.index() for s in poss_stone_maps], dtype=object)
  potion_maps = np.array([p.index(perm_index_to_index) for p in
                          poss_potion_maps], dtype=object)

  # The graph distribution is an unordered mapping, we sort it to make debugging
  # easier and so that we can extract a list of graphs and a list of
  # probabilities for those graphs and this will be consistent across runs.
  graphs_distr = graphs.graph_distr(constraints)
  graphs_distr_as_list = list(graphs_distr.items())
  graphs_distr_constraints = [graphs.constraint_from_graph(k)
                              for k, _ in graphs_distr_as_list]
  graphs_distr_num_constraints = graphs.get_num_constraints(
      graphs_distr_constraints)
  graphs_distr_sorted = sorted(zip(
      graphs_distr_as_list, graphs_distr_num_constraints,
      graphs_distr_constraints), key=lambda x: (x[2], str(x[1])))
  graphs_list = np.frompyfunc(graphs.Graph, 2, 1)(
      np.array([g[0].node_list for g, _, _ in graphs_distr_sorted],
               dtype=object),
      np.array([g[0].edge_list for g, _, _ in graphs_distr_sorted],
               dtype=object))
  graph_index_distr = np.array([g[1] for g, _, _ in graphs_distr_sorted],
                               dtype=object)

  graphs_with_edge = get_graphs_with_edge(graphs_list, index_to_perm_index)

  # A list of the edges which can be present in a bottleneck.
  # i.e. edges of the cube.
  graph_important_edges = graphs.cube_edges()

  # A list of all partial information we could have about a graph given that
  # we have performed some set of stone in potion experiments.
  possible_partial_graph_indices = get_possible_partial_graph_indices(
      graph_important_edges, graphs_list)
  # Map from a simple partial graph index which enumerates all representable
  # graphs to an index into the list of reachable partial graphs.
  partial_graph_index_to_possible_index = {ind: i for i, ind in enumerate(
      possible_partial_graph_indices)}

  # A map which goes from a partial graph to a list of graphs which are
  # possible given the partial info.
  partial_graph_to_matching_graphs = get_partial_graph_to_matching_graphs(
      graphs_list, possible_partial_graph_indices)

  # A map from a partial graph to an updated partial graph given an
  # observation.
  partial_graph_update = get_partial_graph_update(
      graphs_list, graph_important_edges, possible_partial_graph_indices,
      partial_graph_index_to_possible_index)

  # A map from a perceived stone to its reward.
  stone_to_reward = np.array(
      [aligned_stone_from_index(AlignedStoneIndex(i)).reward
       for i in range(stones_and_potions.AlignedStone.num_types)],
      dtype=object)
  # A map from a perceived stone to an index which ignores the reward.
  drop_reward = np.array(
      [aligned_stone_from_index(AlignedStoneIndex(i)).coords_only_index()
       for i in range(stones_and_potions.AlignedStone.num_types)],
      dtype=object)

  # Compute a list of possible outcomes (perceived stones) given a perceived
  # stone which we apply a potion to.
  possible_latent_dims = get_possible_latent_dims(index_to_perm_index)

  poss_p_maps, poss_s_maps = get_poss_potion_maps_and_stone_maps(
      perm_index_to_index)
  possible_latent_dirs = get_possible_latent_dirs(index_to_perm_index)
  partial_potion_map_update = get_partial_potion_map_update(
      index_to_perm_index, perm_index_to_index)
  partial_stone_map_update = get_partial_stone_map_update()

  # For each perceived potion we create a mask on the observed no effect bit
  # field which selects the entries for this potion.
  potion_masks_list = []
  for j in reversed(range(stones_and_potions.PerceivedPotion.num_types)):
    binary_list = ['1' if i == j else '0' for i in range(
        stones_and_potions.PerceivedPotion.num_types)]
    mask = int(''.join([''.join(binary_list) for _ in range(
        stones_and_potions.LatentStone.num_types)]), 2)
    potion_masks_list.append(mask)
  potion_masks = np.array(potion_masks_list, dtype=object)

  perceived_potions = np.array(
      [perceived_potion_from_index(PerceivedPotionIndex(i)) for i in range(
          stones_and_potions.PerceivedPotion.num_types)],
      dtype=object)
  potion_to_pair = np.array(
      [stones_and_potions.PerceivedPotion(
          perceived_dim=p.perceived_dim, perceived_dir=-p.perceived_dir).index()
       for p in perceived_potions], dtype=object)

  edge_exists = get_edge_exists(possible_partial_graph_indices)

  react_result = get_react_result(
      possible_partial_graph_indices, edge_exists, drop_reward)

  missing_edge_no_change = get_missing_edge_no_change(
      index_to_perm_index, graph_important_edges)
  update_partial_graph_no_change = get_update_partial_graph_no_change(
      graphs_list, possible_partial_graph_indices,
      partial_graph_index_to_possible_index, graph_important_edges)

  partial_stone_map_to_stone_map = get_partial_stone_map_to_stone_map()
  no_effect_from_partial_chem = get_no_effect_from_partial_chem(
      index_to_perm_index)

  precomputed = PrecomputedMaps(
      graphs_list=graphs_list, graph_index_distr=graph_index_distr,
      partial_graph_to_matching_graphs=partial_graph_to_matching_graphs,
      partial_graph_update=partial_graph_update,
      stone_to_reward=stone_to_reward, drop_reward=drop_reward,
      partial_graph_index_to_possible_index=(
          partial_graph_index_to_possible_index),
      graphs_with_edge=graphs_with_edge, edge_exists=edge_exists,
      stone_maps=stone_maps, potion_maps=potion_maps,
      possible_latent_dims=possible_latent_dims, poss_p_maps=poss_p_maps,
      poss_s_maps=poss_s_maps, react_result=react_result,
      possible_latent_dirs=possible_latent_dirs,
      partial_potion_map_update=partial_potion_map_update,
      partial_stone_map_update=partial_stone_map_update,
      potion_masks=potion_masks, potion_to_pair=potion_to_pair,
      perm_index_to_index=perm_index_to_index,
      index_to_perm_index=index_to_perm_index,
      missing_edge_no_change=missing_edge_no_change,
      update_partial_graph_no_change=update_partial_graph_no_change,
      partial_stone_map_to_stone_map=partial_stone_map_to_stone_map,
      no_effect_from_partial_chem=no_effect_from_partial_chem
  )

  return precomputed
