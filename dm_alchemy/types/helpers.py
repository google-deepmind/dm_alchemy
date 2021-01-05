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

import math
from typing import List, Sequence

import numpy as np

UNKNOWN = -1000
END_TRIAL = -2


def str_np_array_construct(a: np.ndarray) -> str:
  return 'np.' + repr(a)


def perm_to_index(perm: Sequence[int], perm_index_to_index: np.ndarray) -> int:
  """Converts a permutation to an integer.

  We first treat the permutation as a tuple of integers which can be any value
  between 0 and len(perm) - 1. Then we use the precomputed perm_index_to_index
  to convert from this to indices between 0 and len(perm)!.

  For example if the permutation is [0, 1, 2] this maps to
    0 * 3^2 + 1 * 3^1 + 2 * 3^0 = 5
  Then we look up perm_index_to_index[5] which is 0.

  Args:
    perm: A permutation.
    perm_index_to_index: A matrix which converts valid permutations of length 3
      to indices between 0 and 3!.

  Returns:
    An integer representing the permutation.
  """

  return perm_index_to_index[np.ravel_multi_index(
      tuple(perm), tuple(len(perm) for _ in range(len(perm))))]


def perm_from_index(
    ind: int, num_elements, index_to_perm_index: np.ndarray) -> List[int]:
  # Do the inverse of perm_to_index.
  return [int(i) for i in np.unravel_index(
      index_to_perm_index[ind],
      tuple(num_elements for _ in range(num_elements)))]


def partial_perm_to_index(
    partial_perm: Sequence[int], perm_index_to_index: np.ndarray) -> int:
  """Converts permutation of length 3 with potentially unknown values to an int."""
  # We cannot have just 1 unknown value because knowing the others mean it is
  # determined. Therefore with a length 3 sequence we either have 0, 1 or 3
  # knowns.

  # To make this work for permutations of lengths other than 3 we would have to
  # consider all cases where the number of knowns is 0, 1, .... n - 2, n.
  # If the number of knowns is m there are m! ways to order them, n choose m
  # ways to select the known values and n choose m ways to place them in the
  # permutation. Since we only need to deal with permutations of length 3 we
  # just deal with that special case here.
  if len(partial_perm) != 3:
    raise ValueError('Function only deals with permutations of length 3.')

  first_unknown = UNKNOWN
  first_known = UNKNOWN
  known_val = UNKNOWN
  for i, p in enumerate(partial_perm):
    if p == UNKNOWN:
      if first_unknown == UNKNOWN:
        first_unknown = i
    else:
      if first_known == UNKNOWN:
        first_known = i
        known_val = p
  # If we have 0 unknowns encode as normal.
  if first_unknown == UNKNOWN:
    return perm_to_index(partial_perm, perm_index_to_index)
  num_axes = len(partial_perm)
  num_simple_perms = math.factorial(num_axes)
  # If we have 0 knowns use the next value.
  if first_known == UNKNOWN:
    return num_simple_perms
  # If we have 2 unknowns then we can encode this using the position and value
  # of the first (and only) known element.
  return num_simple_perms + 1 + int(np.ravel_multi_index(
      (first_known, known_val), (num_axes, num_axes)))


def partial_perm_from_index(
    ind: int, num_elements: int, index_to_perm_index: np.ndarray
) -> List[int]:
  """Converts int to permutation of length 3 with potentially unknown values."""
  num_simple_perms = math.factorial(num_elements)
  if ind < num_simple_perms:
    return perm_from_index(ind, num_elements, index_to_perm_index)
  none_known = [UNKNOWN for _ in range(num_elements)]
  if ind == num_simple_perms:
    return none_known
  known_pos, known_val = np.unravel_index(
      ind - num_simple_perms - 1, (num_elements, num_elements))  # pylint: disable=unbalanced-tuple-unpacking
  none_known[known_pos] = int(known_val)
  return none_known
