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
"""Functions for converting to and from precomputed maps protos."""

import base64
import math
import os

from typing import List, Optional, Sequence, Tuple

from dm_alchemy import io
from dm_alchemy.encode import precomputed_maps_pb2
from dm_alchemy.types import graphs
from dm_alchemy.types import stones_and_potions
import numpy as np


def _int_to_string(i: int, bytes_per_int: int) -> str:
  return base64.b64encode(
      i.to_bytes(bytes_per_int, 'big', signed=False)).decode('ascii')


def _string_to_int(s: str) -> int:
  return int.from_bytes(base64.b64decode(s.encode(
      'ascii')), 'big', signed=False)


def _write_to_bin(
    data: np.ndarray, folder: str, name: str, proto_type,
    entries_from_list=lambda l: l
) -> None:
  proto = proto_type(
      entries=entries_from_list(data.ravel().tolist()),
      shape=data.shape)
  io.write_proto(os.path.join(folder, name), proto.SerializeToString())


def write_int_array(data: np.ndarray, folder: str, name: str) -> None:
  _write_to_bin(data, folder, name, precomputed_maps_pb2.IntArray)


def write_float_array(data: np.ndarray, folder: str, name: str) -> None:
  _write_to_bin(data, folder, name, precomputed_maps_pb2.FloatArray)


def _bitfield_entries_from_list(l: Sequence[int]) -> List[str]:
  max_new_list_int = max(l) if l else 0
  num_bytes_needed = math.ceil(max_new_list_int.bit_length() / 8)
  return [_int_to_string(i, num_bytes_needed) for i in l]


def write_bitfield_array(data: np.ndarray, folder: str, name: str) -> None:
  _write_to_bin(
      data, folder, name, precomputed_maps_pb2.BitfieldArray,
      _bitfield_entries_from_list)


def _list_int_entries_from_list(
    l: Sequence[Optional[Sequence[int]]]
) -> List[precomputed_maps_pb2.ListIntsArray.Entry]:
  return [precomputed_maps_pb2.ListIntsArray.Entry(
      list_present=e is not None, entries=e) for e in l]


def write_list_ints_array(data: np.ndarray, folder: str, name: str) -> None:
  _write_to_bin(
      data, folder, name, precomputed_maps_pb2.ListIntsArray,
      _list_int_entries_from_list)


def _possible_latent_dir_entries_from_list(
    l: Sequence[Tuple[bool, Sequence[int]]]
) -> List[precomputed_maps_pb2.PossibleLatentDirs.Entry]:
  return [precomputed_maps_pb2.PossibleLatentDirs.Entry(
      could_be_unchanged=e[0], plausible_latent_dirs=e[1]) for e in l]


def write_possible_latent_dirs(
    data: np.ndarray, folder: str, name: str
) -> None:
  _write_to_bin(
      data, folder, name, precomputed_maps_pb2.PossibleLatentDirs,
      _possible_latent_dir_entries_from_list)


def _partial_potion_map_update_entries_from_list(
    l: Sequence[Tuple[int, int]]
) -> List[int]:
  max_vals = (stones_and_potions.PartialPotionMap.num_axis_assignments + 1,
              stones_and_potions.PartialPotionMap.num_dir_assignments + 1)
  return [int(np.ravel_multi_index((e[0] + 1, e[1] + 1), max_vals)) for e in l]


def write_partial_potion_map_update(
    data: np.ndarray, folder: str, name: str
) -> None:
  _write_to_bin(
      data, folder, name, precomputed_maps_pb2.IntArray,
      _partial_potion_map_update_entries_from_list)


def graph_to_proto(
    graph: graphs.Graph
) -> precomputed_maps_pb2.GraphArray.Entry:
  adj_mat = graphs.convert_graph_to_adj_mat(graph)
  edge_values = graphs.edge_values_from_adj_mat(adj_mat)
  return precomputed_maps_pb2.GraphArray.Entry(edge_present=edge_values)


def _graph_entries_from_list(
    l: Sequence[graphs.Graph]
) -> List[precomputed_maps_pb2.GraphArray.Entry]:
  return [graph_to_proto(graph) for graph in l]


def write_graph_array(data: np.ndarray, folder: str, name: str) -> None:
  _write_to_bin(
      data, folder, name, precomputed_maps_pb2.GraphArray,
      _graph_entries_from_list)


def _load_proto(
    folder: str, name: str, proto_type, proto_entry_to_array_entry=lambda e: e
) -> np.ndarray:
  """Loads serialized proto file representing a numpy array."""
  serialized = io.read_proto(os.path.join(folder, name))
  proto = proto_type.FromString(serialized)
  def pyfunc(i):
    return proto_entry_to_array_entry(proto.entries[i])
  return np.frompyfunc(pyfunc, 1, 1)(np.reshape(np.arange(len(
      proto.entries)), proto.shape))


def load_int_array(folder: str, name: str) -> np.ndarray:
  return _load_proto(folder, name, precomputed_maps_pb2.IntArray)


def load_float_array(folder: str, name: str) -> np.ndarray:
  return _load_proto(folder, name, precomputed_maps_pb2.FloatArray)


def load_bitfield_array(folder: str, name: str) -> np.ndarray:
  return _load_proto(
      folder, name, precomputed_maps_pb2.BitfieldArray, _string_to_int)


def _list_int_proto_to_array(
    entry: precomputed_maps_pb2.ListIntsArray.Entry
) -> Optional[List[int]]:
  return None if not entry.list_present else list(entry.entries)


def load_list_ints_array(folder: str, name: str) -> np.ndarray:
  return _load_proto(
      folder, name, precomputed_maps_pb2.ListIntsArray,
      _list_int_proto_to_array)


def _possible_latent_dir_proto_to_array(
    entry: precomputed_maps_pb2.PossibleLatentDirs.Entry
) -> Tuple[bool, List[int]]:
  return entry.could_be_unchanged, list(entry.plausible_latent_dirs)


def load_possible_latent_dirs(folder: str, name: str) -> np.ndarray:
  return _load_proto(
      folder, name, precomputed_maps_pb2.PossibleLatentDirs,
      _possible_latent_dir_proto_to_array)


def _partial_potion_map_update_proto_to_array(entry: int) -> Tuple[int, int]:
  max_vals = (stones_and_potions.PartialPotionMap.num_axis_assignments + 1,
              stones_and_potions.PartialPotionMap.num_dir_assignments + 1)
  partial_potion_map_index = np.unravel_index(entry, max_vals)
  return partial_potion_map_index[0] - 1, partial_potion_map_index[1] - 1


def load_partial_potion_map_update(folder: str, name: str) -> np.ndarray:
  return _load_proto(
      folder, name, precomputed_maps_pb2.IntArray,
      _partial_potion_map_update_proto_to_array)


def proto_to_graph(
    entry: precomputed_maps_pb2.GraphArray.Entry
) -> graphs.Graph:
  return graphs.convert_adj_mat_to_graph(graphs.adj_mat_from_edge_values(
      entry.edge_present))


def load_graph_array(folder: str, name: str) -> np.ndarray:
  return _load_proto(
      folder, name, precomputed_maps_pb2.GraphArray, proto_to_graph)
