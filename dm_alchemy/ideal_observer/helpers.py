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
"""Helper functions and global variables for ideal observer."""

import itertools
from typing import List, Sequence, Tuple, TypeVar

IMPOSSIBLE = -1001


T = TypeVar('T')


def sorted_intersection(
    list_0: Sequence[T], list_1: Sequence[T]
) -> Tuple[List[T], List[int]]:
  """Finds the intersection and the indices of the remaining elements in list_0.

  Both lists must be sorted in ascending order.

  Args:
    list_0: Sequence of elements which can be compared to each other and the
      elements in list_1.
    list_1: A second sequence of elements.

  Returns:
    A list of the intersection of the two lists.
    A list of the indices into list_0 for the elements in the intersection.
  """
  in_both = []
  still_in_0 = []
  index_0 = 0
  index_1 = 0
  while index_0 < len(list_0) and index_1 < len(list_1):
    value_0 = list_0[index_0]
    value_1 = list_1[index_1]
    if value_0 == value_1:
      still_in_0.append(index_0)
      in_both.append(value_0)
    if value_0 <= value_1:
      index_0 += 1
    if value_1 <= value_0:
      index_1 += 1
  return in_both, still_in_0


def list_to_bitfield(l: Sequence[int]) -> int:
  bitfield = 0
  for i in l:
    bitfield |= (1 << i)
  return bitfield


def bitfield_to_list(b: int) -> List[int]:
  ret = []
  for i in itertools.count():
    mask = 1 << i
    if b & mask:
      ret.append(i)
    if mask > b:
      break
  return ret


def pack_to_bitfield(ints_and_num_bits: Sequence[Tuple[int, int]]) -> int:
  """Packs a sequence of ints into a single int which acts like a bitfield.

  Args:
    ints_and_num_bits: Sequence of tuples each containing an int and the max
      number of bits required to represent that int.

  Returns:
    A single arbitrary precision int with all the passed ints packed into it.
  """
  int_rep = 0
  bits_count = 0
  for sub_int_rep, num_bits in ints_and_num_bits:
    for i in range(num_bits):
      if sub_int_rep & (1 << i):
        int_rep |= (1 << (bits_count + i))
    bits_count += num_bits
  return int_rep
