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
"""Classes and functions for stones and potions used in ideal observer."""

import collections
import copy
import itertools
import math
from typing import Counter, Dict, List, MutableSequence, NewType, Optional, Reversible, Sequence, Tuple

import dataclasses
from dm_alchemy.types import helpers
import numpy as np
from scipy.spatial import transform
import scipy.special


CAULDRON = -1


@dataclasses.dataclass
class RewardWeights:
  """Stores parameters defining the reward of a stone and apply them."""
  coefficients: Sequence[int]
  offset: int
  bonus: int

  def __call__(self, coords: Sequence[int]):
    """Gets value of latent coords, must be a list from set [-1, 1]."""
    node_value = np.dot(coords, self.coefficients) + self.offset
    if np.all([x == 1 for x in coords]):
      node_value += self.bonus
    return node_value


_POSS_DIRS = (-1, 1)
_POSS_AXES = (0, 1, 2)
POSS_REWARDS = (-3, -1, 1, 3)


def get_num_axes() -> int:
  return len(_POSS_AXES)


def get_num_dirs() -> int:
  return len(_POSS_DIRS)


def max_reward() -> int:
  return max(POSS_REWARDS)


def poss_reward_to_index(reward: int) -> int:
  # Possible rewards are -3, -1, 1, 3 convert this to 0, 1, 2, 3.
  return (reward + 3) // 2


def index_to_poss_reward(ind: int) -> int:
  # Inverse of poss_reward_to_index.
  return (ind * 2) - 3


def dir_to_index(direction: int) -> int:
  # Possible directions are -1, 1 convert this to 0, 1.
  return (direction + 1) // 2


def index_to_dir(ind: int) -> int:
  # Inverse of dir_to_index.
  return (ind * 2) - 1


def unknown_dir_to_index(direction: int) -> int:
  # Possible directions are -1, 1, helpers.UNKNOWN convert this to 0, 1, 2.
  if direction == helpers.UNKNOWN:
    return len(_POSS_DIRS)
  return dir_to_index(direction)


def index_to_unknown_dir(ind: int) -> int:
  # Inverse of unknown_dir_to_index.
  if ind == len(_POSS_DIRS):
    return helpers.UNKNOWN
  return index_to_dir(ind)


def coords_to_index(coords: Reversible[int]) -> int:
  # Possible directions are -1, 1 convert this to 0, 1.
  # Coords are reversed to match the way unity converts between coords and
  # indices.
  return int(np.ravel_multi_index(
      tuple(dir_to_index(c) for c in reversed(coords)),
      tuple(len(_POSS_DIRS) for _ in _POSS_AXES)))


def index_to_coords(ind: int) -> np.ndarray:
  # Inverse of coords_to_index.
  return np.array([index_to_dir(int(i)) for i in reversed(np.unravel_index(
      ind, tuple(len(_POSS_DIRS) for _ in _POSS_AXES)))], object)


def get_all_dim_ordering() -> List[Tuple[int, ...]]:
  # return a sorted list of all possible orderings of axes
  all_dim_ordering = sorted(set(itertools.permutations(_POSS_AXES)))
  return all_dim_ordering

AlignedStoneIndex = NewType('AlignedStoneIndex', int)


class AlignedStone:
  """The perceived stone with perceptual features rotated to align with axes."""

  # The number of possible perceived stones 32: 2 possible coords on each of 3
  # dimensions and 4 possible rewards.
  num_dir_assignments = len(_POSS_DIRS) ** len(_POSS_AXES)
  num_types = len(POSS_REWARDS) * num_dir_assignments
  num_bits = math.ceil(math.log2(num_types))
  # For space efficiency in the ideal observer we do not allow more than 3
  # stones per trial.
  max_present = 3
  count_num_bits = math.ceil(math.log2(max_present))

  def __init__(self, reward: int, coords: np.ndarray):
    self.reward = reward
    self.aligned_coords = coords

  def __hash__(self) -> int:
    return hash((self.reward, tuple(self.aligned_coords)))

  def __eq__(self, other: 'AlignedStone') -> bool:
    return (self.reward == other.reward and
            np.array_equal(self.aligned_coords, other.aligned_coords))

  def __repr__(self) -> str:
    return ('AlignedStone(reward={reward}, '
            'coords={coords})'.format(
                reward=self.reward,
                coords=helpers.str_np_array_construct(self.aligned_coords)))

  def index(self) -> AlignedStoneIndex:
    coords_index = self.coords_only_index()
    reward_index = poss_reward_to_index(self.reward)
    return AlignedStoneIndex(np.ravel_multi_index(
        (coords_index, reward_index),
        (AlignedStone.num_dir_assignments, len(POSS_REWARDS))))

  def coords_only_index(self) -> int:
    return coords_to_index(self.aligned_coords)

  def swap_dims(self, dim_map: Sequence[int]) -> None:
    aligned_coords = np.copy(self.aligned_coords)
    for i, new_dim in enumerate(dim_map):
      self.aligned_coords[new_dim] = aligned_coords[i]


def aligned_stone_from_index(ind: AlignedStoneIndex) -> AlignedStone:
  coords_index, reward_index = np.unravel_index(
      ind, (AlignedStone.num_dir_assignments, len(POSS_REWARDS)))
  coords = index_to_coords(int(coords_index))
  reward = index_to_poss_reward(int(reward_index))
  return AlignedStone(reward, coords)


def possible_aligned_stones() -> List[AlignedStone]:
  return [aligned_stone_from_index(AlignedStoneIndex(i))
          for i in range(AlignedStone.num_types)]


def possible_aligned_stones_coords_only() -> List[AlignedStone]:
  return [AlignedStone(-3, stone.latent_coords)
          for stone in possible_latent_stones()]


def random_aligned_stone(
    random_state: np.random.RandomState) -> AlignedStone:
  return aligned_stone_from_index(
      random_state.randint(0, AlignedStone.num_types))


def aligned_stones_with_coords(
    coords: 'LatentStone') -> List[AlignedStone]:
  return [AlignedStone(reward, coords.latent_coords)
          for reward in POSS_REWARDS]


class PerceivedStone:
  """The perceived reward and coordinates (in perceptual space) of a stone."""

  def __init__(self, reward: int, coords: np.ndarray):
    self.reward = reward
    self.perceived_coords = coords

  def __hash__(self) -> int:
    return hash((self.reward, tuple(self.perceived_coords)))

  def __eq__(self, other: 'PerceivedStone') -> bool:
    return (self.reward == other.reward and
            np.array_equal(self.perceived_coords, other.perceived_coords))

  def __repr__(self) -> str:
    return ('PerceivedStone(reward={reward}, '
            'coords={coords})'.format(
                reward=self.reward,
                coords=helpers.str_np_array_construct(self.perceived_coords)))


def align_coords(
    perceived_stone: PerceivedStone, rotation: np.ndarray
) -> np.ndarray:
  return np.matmul(rotation, perceived_stone.perceived_coords.astype(int))


def aligns(
    perceived_stone: PerceivedStone, rotation: np.ndarray
) -> Tuple[bool, np.ndarray]:
  coords = align_coords(perceived_stone, rotation)
  # Valid if all coords are close to 1 or -1.
  valid = all(abs(abs(apc) - 1.0) < 0.0001 for apc in coords)
  return valid, coords


def aligned_stone_from_coords(coords: np.ndarray, reward: int) -> AlignedStone:
  coords = np.array([
      int(round(i)) for i in coords], dtype=object)
  return AlignedStone(reward, coords)


def align(
    perceived_stone: PerceivedStone, rotation: np.ndarray
) -> AlignedStone:
  """Gets aligned stone from perceived stone given the rotation."""
  valid, coords = aligns(perceived_stone, rotation)
  if not valid:
    raise ValueError('Rotation passed does not align stone.')
  return aligned_stone_from_coords(coords, perceived_stone.reward)


def unalign(
    aligned_stone: AlignedStone, rotation: np.ndarray
) -> PerceivedStone:
  """Get perceived stone from aligned stone given the rotation."""
  perceived_coords = np.matmul(
      np.linalg.inv(rotation), aligned_stone.aligned_coords.astype(int))
  # All coords should be close to 1, 0 or -1.
  if not all(abs(apc) < 0.0001 or abs(abs(apc) - 1.0) < 0.0001
             for apc in perceived_coords):
    raise ValueError('Rotation does not take aligned stone to integer coords.')
  perceived_coords = np.array([
      int(round(i)) for i in perceived_coords], dtype=object)
  return PerceivedStone(aligned_stone.reward, perceived_coords)


def get_new_mapping_to_old_mapping(
    prev_rotation: np.ndarray, new_rotation: np.ndarray,
    observed_stone: AlignedStone
) -> 'StoneMap':
  """Maps aligned stones under previous rotation to aligned stones under new rotation."""
  # Apply inv new_rotation to aligned coords to get the new set of perceived
  # coords, then apply prev_rotation to these, coordinates that end up valid
  # could have been learned about so need to go into the mapping.
  perceived_stone = unalign(observed_stone, prev_rotation)
  new_aligned = align(perceived_stone, new_rotation)
  return StoneMap(np.where(observed_stone.aligned_coords ==
                           new_aligned.aligned_coords, 1, -1))


def rotation_from_angles(angles: Sequence[float]) -> np.ndarray:
  """Gets rotation matrix from list of angles, scaling as required."""
  if not any(angles):
    return np.eye(3, dtype=int)
  rotation = transform.Rotation.from_euler(
      'xyz', angles, degrees=True).as_matrix()
  scale = np.diag([1.0 if angle else math.sqrt(2) for angle in angles])
  transformation = np.matmul(scale, rotation)
  if not all(abs(c) < 0.0001 or abs(abs(c) - 1.0) < 0.0001
             for c in transformation.reshape((-1,))):
    raise ValueError(
        'Transformation should be all -1, 0 or 1 but is ' + str(transformation)
        + ' for angles ' + str(angles))
  return transformation.astype(int)


def rotation_to_angles(rotation: np.ndarray) -> Sequence[float]:
  # First scale it.
  column_norm = np.linalg.norm(rotation, axis=0)
  s = np.diag(1 / column_norm)
  rotation = np.matmul(s, rotation)
  return transform.Rotation.from_matrix(rotation).as_euler(
      'xyz', degrees=True).tolist()


def possible_rotations() -> List[np.ndarray]:
  list_angles = [[0, 0, 0], [0, 0, -45], [0, -45, 0], [-45, 0, 0]]
  return [rotation_from_angles(angles) for angles in list_angles]


def random_rotation(random_state: np.random.RandomState):
  poss_rotations = possible_rotations()
  return poss_rotations[random_state.choice(len(poss_rotations))]


def rotations_equal(rotation1: np.ndarray, rotation2: np.ndarray):
  return rotation_to_angles(rotation1) == rotation_to_angles(rotation2)


LatentStoneIndex = NewType('LatentStoneIndex', int)


class LatentStone:
  """The latent coordinates of a stone."""

  # The number of possible latent stones 8: 2 possible coords on each of 3
  # dimensions.
  num_types = len(_POSS_DIRS) ** len(_POSS_AXES)
  num_bits = math.ceil(math.log2(num_types))

  def __init__(self, coords: np.ndarray):
    self.latent_coords = coords

  def reward(self) -> int:
    return int(sum(self.latent_coords))

  def __hash__(self) -> int:
    return hash(tuple(self.latent_coords))

  def __eq__(self, other: 'LatentStone') -> bool:
    return np.array_equal(self.latent_coords, other.latent_coords)

  def __repr__(self) -> str:
    return 'LatentStone(coords={coords})'.format(
        coords=helpers.str_np_array_construct(self.latent_coords))

  def index(self) -> LatentStoneIndex:
    return LatentStoneIndex(coords_to_index(self.latent_coords))


def latent_stone_from_index(ind: LatentStoneIndex) -> LatentStone:
  return LatentStone(index_to_coords(ind))


def random_latent_stone(random_state: np.random.RandomState) -> LatentStone:
  return latent_stone_from_index(
      random_state.randint(0, LatentStone.num_types))


def possible_latent_stones() -> List[LatentStone]:
  return [latent_stone_from_index(LatentStoneIndex(ind))
          for ind in range(LatentStone.num_types)]


StoneMapIndex = NewType('StoneMapIndex', int)


class StoneMap:
  """A map that takes stones from perceptual to latent coordinates."""

  # The number of possible stone maps 8: (2 possible directions on each of 3
  # dimensions). The assignment of perceived axes to latent axes can be chosen
  # arbitrarily given that all oriented graphs and assignment of potions to
  # dimensions and directions are considered.
  num_types = len(_POSS_DIRS) ** len(_POSS_AXES)
  num_bits = math.ceil(math.log2(num_types))

  def __init__(self, pos_dir: np.ndarray):
    self.latent_pos_dir = pos_dir

  def __hash__(self) -> int:
    return hash(tuple(self.latent_pos_dir))

  def __eq__(self, other: 'StoneMap') -> bool:
    return np.array_equal(self.latent_pos_dir, other.latent_pos_dir)

  def __repr__(self) -> str:
    return ('StoneMap(pos_dir={pos_dir})'.format(
        pos_dir=helpers.str_np_array_construct(self.latent_pos_dir)))

  def apply(self, stone: AlignedStone) -> LatentStone:
    return LatentStone(np.where(stone.aligned_coords ==
                                self.latent_pos_dir, 1, -1))

  def apply_inverse(self, stone: LatentStone) -> AlignedStone:
    # Map is actually equal to its inverse
    return AlignedStone(stone.reward(), np.where(
        stone.latent_coords == self.latent_pos_dir, 1, -1))

  def apply_to_potion(self, potion: 'LatentPotion') -> 'LatentPotion':
    return LatentPotion(
        potion.latent_dim,
        potion.latent_dir * self.latent_pos_dir[potion.latent_dim])

  def index(self) -> StoneMapIndex:
    return StoneMapIndex(coords_to_index(self.latent_pos_dir))

  def swap_dims(self, dim_map: Sequence[int]) -> None:
    latent_pos_dir = np.copy(self.latent_pos_dir)
    for i, new_dim in enumerate(dim_map):
      self.latent_pos_dir[new_dim] = latent_pos_dir[i]

  def chain(self, new_to_old: 'StoneMap') -> None:
    # Combine this map with the one passed in.
    new_to_reward = np.where(
        self.latent_pos_dir == new_to_old.latent_pos_dir, 1, -1)
    self.latent_pos_dir = new_to_reward

  def consistent_with_stones(
      self, aligned_stones: Sequence[AlignedStone]
  ) -> bool:
    for aligned_stone in aligned_stones:
      if self.apply(aligned_stone).reward() != aligned_stone.reward:
        return False
    return True


def stone_map_from_index(ind: StoneMapIndex) -> StoneMap:
  return StoneMap(index_to_coords(ind))


def random_stone_map(random_state: np.random.RandomState) -> StoneMap:
  return stone_map_from_index(random_state.randint(0, StoneMap.num_types))


def possible_stone_maps() -> List[StoneMap]:
  return [stone_map_from_index(StoneMapIndex(i))
          for i in range(StoneMap.num_types)]


def stone_map_distr(
    stone_maps: Sequence[StoneMapIndex]
) -> Dict[StoneMapIndex, float]:
  """Returns a distribution over possible stone maps."""
  # I think it is valid to assume the latent space axis corresponds directly to
  # the perceptual space axis since we do not make this assumption for potion
  # colours.
  return {s: 1 / len(stone_maps) for s in stone_maps}


def all_fixed_stone_map() -> StoneMap:
  return StoneMap(pos_dir=np.array([1, 1, 1]))


PartialStoneMapIndex = NewType('PartialStoneMapIndex', int)


class PartialStoneMap(StoneMap):
  """Partial info on the map from stone perceptual space to latent space."""

  # The number of possible partial stone maps 27: (3 possible directions
  # including unknown on each of 3 dimensions).
  num_types = (len(_POSS_DIRS) + 1) ** len(_POSS_AXES)
  num_bits = math.ceil(math.log2(num_types))

  def matches(self, other: StoneMap) -> bool:
    return all(d == helpers.UNKNOWN or d == other_d for d, other_d
               in zip(self.latent_pos_dir, other.latent_pos_dir))

  def fill_gaps(self) -> List[StoneMap]:
    """Returns all stone maps possible given the partial information."""
    return [StoneMap(np.array(m)) for m in itertools.product(
        *[_POSS_DIRS if d == helpers.UNKNOWN else [d]
          for d in self.latent_pos_dir])]

  def update(self, axis: int, pos_dir: int) -> None:
    self.latent_pos_dir = tuple(d if i != axis else pos_dir
                                for i, d in enumerate(self.latent_pos_dir))

  def index(self) -> PartialStoneMapIndex:
    return PartialStoneMapIndex(np.ravel_multi_index(
        tuple(unknown_dir_to_index(d) for d in self.latent_pos_dir),
        tuple(len(_POSS_DIRS) + 1 for _ in self.latent_pos_dir)))


def partial_stone_map_from_index(ind: PartialStoneMapIndex) -> PartialStoneMap:
  pos_dir_indices = np.unravel_index(
      ind, tuple(len(_POSS_DIRS) + 1 for _ in _POSS_AXES))
  pos_dir = np.array(
      [index_to_unknown_dir(int(d)) for d in pos_dir_indices], object)
  return PartialStoneMap(pos_dir)


def partial_stone_map_from_possibles(
    possibles: Sequence[StoneMapIndex]) -> PartialStoneMap:
  all_poss = np.stack((stone_map_from_index(s).latent_pos_dir
                       for s in possibles))
  uniques = [np.unique(all_poss[:, i]) for i in range(get_num_axes())]
  return PartialStoneMap(np.array([u[0] if len(u) == 1 else helpers.UNKNOWN
                                   for u in uniques]))


def partial_stone_map_from_single_obs(
    axis: int, pos_dir: int) -> PartialStoneMap:
  return PartialStoneMap(np.array([pos_dir if i == axis else helpers.UNKNOWN
                                   for i in _POSS_AXES]))


def possible_partial_stone_maps() -> List[PartialStoneMap]:
  return [partial_stone_map_from_index(PartialStoneMapIndex(i))
          for i in range(PartialStoneMap.num_types)]


PerceivedPotionIndex = NewType('PerceivedPotionIndex', int)


class PerceivedPotion:
  """The perceived dimension and direction (in perceptual space) of a potion."""

  # The number of possible perceived potions 6: one per axis and direction.
  num_types = len(_POSS_DIRS) * len(_POSS_AXES)
  num_bits = math.ceil(math.log2(num_types))
  # For space efiiciency in the ideal observer we do not allow more than 12
  # potions per trial.
  max_present = 12
  count_num_bits = math.ceil(math.log2(max_present))

  def __init__(self, perceived_dim: int, perceived_dir: int):
    self.perceived_dim = perceived_dim
    self.perceived_dir = perceived_dir

  def __hash__(self) -> int:
    return hash((self.perceived_dim, self.perceived_dir))

  def __eq__(self, other: 'PerceivedPotion') -> bool:
    return (self.perceived_dim == other.perceived_dim and
            self.perceived_dir == other.perceived_dir)

  def __repr__(self) -> str:
    return ('PerceivedPotion(perceived_dim={perceived_dim}, '
            'perceived_dir={perceived_dir})'.format(
                perceived_dim=self.perceived_dim,
                perceived_dir=self.perceived_dir))

  def index(self) -> PerceivedPotionIndex:
    return PerceivedPotionIndex(np.ravel_multi_index(
        (self.perceived_dim, dir_to_index(self.perceived_dir)),
        (len(_POSS_AXES), len(_POSS_DIRS))))


def perceived_potion_from_index(ind: PerceivedPotionIndex) -> PerceivedPotion:
  perceived_dim, perceived_dir = np.unravel_index(
      ind, (len(_POSS_AXES), len(_POSS_DIRS)))
  return PerceivedPotion(int(perceived_dim), index_to_dir(int(perceived_dir)))


def possible_perceived_potions() -> List[PerceivedPotion]:
  return [perceived_potion_from_index(PerceivedPotionIndex(i))
          for i in range(PerceivedPotion.num_types)]


def random_perceived_potion(
    random_state: np.random.RandomState) -> PerceivedPotion:
  return perceived_potion_from_index(
      random_state.randint(0, PerceivedPotion.num_types))


LatentPotionIndex = NewType('LatentPotionIndex', int)


class LatentPotion:
  """The latent space dimension and direction of a potion."""

  # The number of possible latent potions 6: one per axis and direction.
  num_types = len(_POSS_DIRS) * len(_POSS_AXES)
  num_bits = math.ceil(math.log2(num_types))

  def __init__(self, latent_dim: int, latent_dir: int):
    self.latent_dim = latent_dim
    self.latent_dir = latent_dir

  def __hash__(self) -> int:
    return hash((self.latent_dim, self.latent_dir))

  def __eq__(self, other: 'LatentPotion') -> bool:
    return (self.latent_dim == other.latent_dim and
            self.latent_dir == other.latent_dir)

  def __repr__(self) -> str:
    return ('LatentPotion(latent_dim={latent_dim}, latent_dir={latent_dir})'.
            format(latent_dim=self.latent_dim, latent_dir=self.latent_dir))

  def index(self) -> LatentPotionIndex:
    return LatentPotionIndex(np.ravel_multi_index(
        (self.latent_dim, dir_to_index(self.latent_dir)),
        (len(_POSS_AXES), len(_POSS_DIRS))))


def latent_potion_from_index(ind: LatentPotionIndex) -> LatentPotion:
  latent_dim, latent_dir = np.unravel_index(
      ind, (len(_POSS_AXES), len(_POSS_DIRS)))
  return LatentPotion(int(latent_dim), index_to_dir(int(latent_dir)))


def random_latent_potion(random_state: np.random.RandomState) -> LatentPotion:
  return latent_potion_from_index(
      random_state.randint(0, LatentPotion.num_types))


def possible_latent_potions() -> List[LatentPotion]:
  return [latent_potion_from_index(LatentPotionIndex(ind))
          for ind in range(LatentPotion.num_types)]


PotionMapIndex = NewType('PotionMapIndex', int)


class PotionMap:
  """A map that takes potions from perceptual to latent coordinates."""

  # The number of possible potion maps - 6 permutations of the 3 axes and 2
  # possible directions on each of the 3 axes.
  num_axis_assignments = math.factorial(len(_POSS_AXES))
  num_dir_assignments = len(_POSS_DIRS) ** len(_POSS_AXES)
  num_types = num_axis_assignments * num_dir_assignments
  num_bits = math.ceil(math.log2(num_types))

  def __init__(self, dim_map: Sequence[int], dir_map: Sequence[int]):
    # We may update these so specify that they are mutable.
    self.dim_map = dim_map  # type: MutableSequence[int]
    self.dir_map = dir_map  # type: MutableSequence[int]

  def __hash__(self) -> int:
    return hash((tuple(self.dim_map), tuple(self.dir_map)))

  def __eq__(self, other: 'PotionMap') -> bool:
    return self.dim_map == other.dim_map and self.dir_map == other.dir_map

  def __repr__(self) -> str:
    return 'PotionMap(dim_map={dim_map}, dir_map={dir_map})'.format(
        dim_map=self.dim_map, dir_map=self.dir_map)

  def apply(self, potion: PerceivedPotion) -> LatentPotion:
    latent_dim = self.dim_map[potion.perceived_dim]
    latent_dir = 1 if self.dir_map[latent_dim] == potion.perceived_dir else -1
    return LatentPotion(latent_dim, latent_dir)

  def apply_inverse(self, potion: LatentPotion) -> PerceivedPotion:
    if self.dir_map[potion.latent_dim] == potion.latent_dir:
      perceived_dir = 1
    else:
      perceived_dir = -1
    inverse_dim_map = [0, 0, 0]
    for i, d in enumerate(self.dim_map):
      inverse_dim_map[d] = i
    perceived_dim = inverse_dim_map[potion.latent_dim]
    return PerceivedPotion(perceived_dim, perceived_dir)

  def index(self, perm_index_to_index: np.ndarray) -> PotionMapIndex:
    dim_index = helpers.perm_to_index(self.dim_map, perm_index_to_index)
    dir_index = coords_to_index(self.dir_map)
    return PotionMapIndex(np.ravel_multi_index(
        (dim_index, dir_index),
        (PotionMap.num_axis_assignments, PotionMap.num_dir_assignments)))


def potion_map_from_index(
    ind: PotionMapIndex, index_to_perm_index: np.ndarray) -> PotionMap:
  dim_map_index, dir_map_index = np.unravel_index(
      ind, (PotionMap.num_axis_assignments, PotionMap.num_dir_assignments))
  dim_map = helpers.perm_from_index(
      int(dim_map_index), len(_POSS_AXES), index_to_perm_index)
  dir_map = index_to_coords(int(dir_map_index)).tolist()
  return PotionMap(dim_map, dir_map)


def random_potion_map(
    index_to_perm_index: np.ndarray, random_state: np.random.RandomState
) -> PotionMap:
  return potion_map_from_index(random_state.randint(0, PotionMap.num_types),
                               index_to_perm_index)


def possible_potion_maps(
    index_to_perm_index: np.ndarray) -> List[PotionMap]:
  return [potion_map_from_index(PotionMapIndex(i), index_to_perm_index)
          for i in range(PotionMap.num_types)]


def potion_map_distr(
    potion_maps: np.ndarray) -> Dict[PotionMapIndex, float]:
  """Makes a uniform distribution over the possible potion maps."""
  return {p: 1 / len(potion_maps) for p in potion_maps}


def all_fixed_potion_map() -> PotionMap:
  return PotionMap(dim_map=[0, 1, 2], dir_map=[1, 1, 1])


PartialPotionMapIndex = Tuple[int, int]


class PartialPotionMap(PotionMap):
  """Partial info on the map from potion perceptual space to latent space."""

  # The number of possible partial potion maps - 6 normal permutations of the 3
  # axes plus 3 ways to select 1 known axis and 3 ways to place it and 3
  # possible directions on each of the 3 axes.
  num_axis_assignments = sum(
      math.factorial(k) * (int(scipy.special.comb(len(_POSS_AXES), k)) ** 2)
      for k in itertools.chain(range(len(_POSS_AXES) - 1), [len(_POSS_AXES)]))
  num_dir_assignments = (len(_POSS_DIRS) + 1) ** len(_POSS_AXES)
  num_types = num_axis_assignments * num_dir_assignments
  num_bits_axis = math.ceil(math.log2(num_axis_assignments))
  num_bits_dir = math.ceil(math.log2(num_dir_assignments))

  def can_map(self, potion: PerceivedPotion) -> bool:
    latent_dim = self.dim_map[potion.perceived_dim]
    if latent_dim == helpers.UNKNOWN:
      return False
    return self.dir_map[latent_dim] != helpers.UNKNOWN

  def matches(self, other: PotionMap) -> bool:
    return (all(d == helpers.UNKNOWN or d == other_d for d, other_d
                in zip(self.dim_map, other.dim_map)) and
            all(d == helpers.UNKNOWN or d == other_d for d, other_d
                in zip(self.dir_map, other.dir_map)))

  def fill_gaps(self) -> List[PotionMap]:
    """Gets all potion maps possible given this partial info."""
    set_vals = {d for d in self.dim_map if d != helpers.UNKNOWN}
    dims_to_set = [i for i, d in enumerate(self.dim_map)
                   if d == helpers.UNKNOWN]
    remaining_vals = {i for i in _POSS_AXES}.difference(set_vals)
    new_dim_maps = []
    for orders in itertools.permutations(remaining_vals):
      new_dim_map = copy.deepcopy(self.dim_map)
      for i, val in zip(dims_to_set, orders):
        new_dim_map[i] = val
      new_dim_maps.append(new_dim_map)

    new_dir_maps = itertools.product(*[
        _POSS_DIRS if d == helpers.UNKNOWN else [d] for d in self.dir_map])
    return [PotionMap(dim_map, dir_map) for dim_map, dir_map in
            itertools.product(new_dim_maps, new_dir_maps)]

  def update(
      self, perceived_axis: int, latent_axis: int, perceived_dir: int,
      reward_dir: int) -> None:
    """Updates the potion map given the observation.

    Args:
      perceived_axis: The perceived axis of the potion.
      latent_axis: The latent axis the potion acts on.
      perceived_dir: Perceived direction of the potion.
      reward_dir: The direction in which the reward changes.
    """
    # The reward dir shows the direction that the potion moves in latent space.
    if reward_dir == 0:
      # We cannot update anything on the partial mapping but will update the
      # combos
      pass
    else:
      self.dim_map[perceived_axis] = latent_axis
      self.deduce_dim_map_gaps()

      self.dir_map[latent_axis] = 1 if perceived_dir == reward_dir else -1

  def deduce_dim_map_gaps(self) -> None:
    """Updates the dimension map by deducing the last dimension if only 1 left."""
    unknown_dims = [i for i, d in enumerate(self.dim_map)
                    if d == helpers.UNKNOWN]
    if len(unknown_dims) == 1:
      unused_dims = set(_POSS_AXES).difference(
          {d for d in self.dim_map if d != helpers.UNKNOWN})
      # For consistent observations this should always be true but when
      # precomputing maps we can consider inconsistent observations. In this
      # case it doesn't matter what we do.
      if len(unused_dims) == 1:
        self.dim_map[unknown_dims[0]] = unused_dims.pop()

  def index(self, perm_index_to_index) -> PartialPotionMapIndex:
    # Use 2 part index for this otherwise maybe it gets too big
    return (helpers.partial_perm_to_index(self.dim_map, perm_index_to_index),
            int(np.ravel_multi_index(
                tuple(unknown_dir_to_index(d) for d in self.dir_map),
                tuple(len(_POSS_DIRS) + 1 for _ in self.dir_map))))

  def possible_latent_dims(
      self, perceived_potion: PerceivedPotion) -> List[int]:
    """Returns a list of latent space dimensions the potion could act on."""
    # If we don't know which dimension the potion operates on then it could be
    # any that we haven't mapped to another dimension.
    if self.dim_map[perceived_potion.perceived_dim] == helpers.UNKNOWN:
      unused_dims = set(range(3)).difference(set(self.dim_map))
      return sorted(list(unused_dims))
    # If we do know the dimension just return it.
    return [self.dim_map[perceived_potion.perceived_dim]]


def partial_potion_map_from_index(
    ind: PartialPotionMapIndex, index_to_perm_index) -> PartialPotionMap:
  dim_map = helpers.partial_perm_from_index(
      ind[0], get_num_axes(), index_to_perm_index)
  dir_map_indices = np.unravel_index(
      ind[1], tuple(len(_POSS_DIRS) + 1 for _ in _POSS_AXES))
  dir_map = [index_to_unknown_dir(int(d)) for d in dir_map_indices]
  return PartialPotionMap(dim_map, dir_map)


def no_knowledge_partial_potion_map() -> PartialPotionMap:
  return PartialPotionMap([helpers.UNKNOWN, helpers.UNKNOWN, helpers.UNKNOWN],
                          [helpers.UNKNOWN, helpers.UNKNOWN, helpers.UNKNOWN])


def partial_potion_map_from_possibles(
    possibles: Sequence[PotionMapIndex], index_to_perm_index: np.ndarray
) -> PartialPotionMap:
  """Creates a partial potion map from a list of all possible potion maps."""
  potion_maps = [potion_map_from_index(p, index_to_perm_index) for p in
                 possibles]
  poss_dim_map = np.stack((p.dim_map for p in potion_maps))
  poss_dir_map = np.stack((p.dir_map for p in potion_maps))
  unique_dims = [np.unique(poss_dim_map[:, i]) for i in range(get_num_axes())]
  unique_dirs = [np.unique(poss_dir_map[:, i]) for i in range(get_num_axes())]
  ret = PartialPotionMap(
      [u[0] if len(u) == 1 else helpers.UNKNOWN for u in unique_dims],
      [u[0] if len(u) == 1 else helpers.UNKNOWN for u in unique_dirs])
  ret.deduce_dim_map_gaps()
  return ret


def one_obs_partial_potion_map(
    perceived_axis: int, latent_axis: int, perceived_dir: int, reward_dir: int
) -> PartialPotionMap:
  dim_map = [latent_axis if i == perceived_axis else helpers.UNKNOWN
             for i in _POSS_AXES]
  dir_map = [1 if i == latent_axis and perceived_dir == reward_dir else -1
             if i == latent_axis else helpers.UNKNOWN for i in _POSS_AXES]
  return PartialPotionMap(dim_map, dir_map)


def one_obs_possible_potion_maps(
    perceived_axis: int, latent_axis: int, perceived_dir: int, reward_dir: int,
    perm_index_to_index: np.ndarray) -> List[int]:
  return sorted(
      [p.index(perm_index_to_index) for p in one_obs_partial_potion_map(
          perceived_axis, latent_axis, perceived_dir, reward_dir).fill_gaps()])


def one_obs_possible_stone_maps(axis: int, pos_dir: int) -> List[int]:
  return sorted([s.index() for s in partial_stone_map_from_single_obs(
      axis, pos_dir).fill_gaps()])


def one_action_outcome(
    stone: AlignedStone, potion: PerceivedPotion, result: AlignedStone,
    perm_index_to_index: np.ndarray
) -> Tuple[Optional[List[int]], Optional[List[int]]]:
  """Computes possible potion and stones maps given the observation."""
  reward_dir = result.reward - stone.reward
  # If we observe a change we can deduce information
  if reward_dir == 2 or reward_dir == -2:
    stone_diff = result.aligned_coords - stone.aligned_coords
    change_axes = np.where(stone_diff)[0]
    if len(change_axes) != 1:
      return None, None
    stone_axis = change_axes[0]
    stone_dir = stone_diff[stone_axis]
    return (one_obs_possible_potion_maps(
        potion.perceived_dim, stone_axis, potion.perceived_dir,
        reward_dir // 2, perm_index_to_index),
            one_obs_possible_stone_maps(
                stone_axis, 1 if stone_dir == reward_dir else -1))
  return None, None


def update_partial_potion_map(
    stone: AlignedStone, potion: PerceivedPotion,
    result_stone: AlignedStone, partial_potion_map: PartialPotionMap,
    perm_index_to_index: np.ndarray) -> PartialPotionMapIndex:
  """Returns updated partial potion map index given the observation."""
  reward_dir = result_stone.reward - stone.reward
  # If we observe a change we can deduce information
  if reward_dir == 2 or reward_dir == -2:
    stone_diff = result_stone.aligned_coords - stone.aligned_coords
    change_axes = np.where(stone_diff)[0]
    if len(change_axes) != 1:
      return -1, -1
    stone_axis = change_axes[0]
    partial_potion_map.update(
        potion.perceived_dim, stone_axis, potion.perceived_dir,
        reward_dir // 2)
    return partial_potion_map.index(perm_index_to_index)
  return -1, -1


def update_partial_stone_map(
    stone: AlignedStone, result_stone: AlignedStone,
    partial_stone_map: PartialStoneMap) -> PartialStoneMapIndex:
  """Updates the partial stone map info given the observation."""
  reward_dir = result_stone.reward - stone.reward
  # If we observe a change we can deduce information
  if reward_dir == 2 or reward_dir == -2:
    stone_diff = result_stone.aligned_coords - stone.aligned_coords
    change_axes = np.where(stone_diff)[0]
    if len(change_axes) != 1:
      return PartialStoneMapIndex(-1)
    stone_axis = change_axes[0]
    stone_dir = stone_diff[stone_axis]
    partial_stone_map.update(stone_axis, 1 if stone_dir == reward_dir else -1)
    return partial_stone_map.index()
  return PartialStoneMapIndex(-1)


def aligned_stone_indices(
    aligned_stones: Counter[AlignedStone]) -> Counter[AlignedStoneIndex]:
  return collections.Counter({k.index(): v for k, v in aligned_stones.items()})


def perceived_potion_indices(
    perceived_potions: Counter[PerceivedPotion], perm_index_to_index: np.ndarray
) -> Counter[PerceivedPotionIndex]:
  return collections.Counter({k.index(perm_index_to_index): v
                              for k, v in perceived_potions.items()})


def react(
    perceived_stone: AlignedStone, latent_dim: int, latent_dir: int
) -> AlignedStone:
  """Possible outcome without knowing the direction mapping if stone changes."""
  new_reward = max(min(POSS_REWARDS), min(
      max(POSS_REWARDS), perceived_stone.reward + (2 * latent_dir)))
  new_coords = np.copy(perceived_stone.aligned_coords)
  new_coords[latent_dim] = -perceived_stone.aligned_coords[latent_dim]
  return AlignedStone(new_reward, new_coords)


def possible_latent_dirs_and_stone_dirs(
    perceived_potion: PerceivedPotion, latent_dim: int,
    partial_potion_map: PartialPotionMap, partial_stone_map: PartialStoneMap
) -> List[Tuple[int, int]]:
  """Possible latent space directions and stone space directions for a potion.

  Given a perceived potion and an assumed latent space dimension and given what
  we know about the maps from potion space and stone space to latent space, get
  a list of possible latent space directions and stone space directions.

  Args:
    perceived_potion: The potion we are considering.
    latent_dim: The latent dimension the potion applies to.
    partial_potion_map: What we know about how potions map to latent space.
    partial_stone_map: What we know about how stones map to latent space.

  Returns:
    List of 2-element tuples where the first element is a latent space direction
    and the second element is a stone space direction.
  """
  if partial_potion_map.dir_map[latent_dim] == helpers.UNKNOWN:
    all_latent_dirs = _POSS_DIRS
  elif partial_potion_map.dir_map[latent_dim] == 1:
    all_latent_dirs = [perceived_potion.perceived_dir]
  else:
    all_latent_dirs = [-perceived_potion.perceived_dir]
  latent_dirs_stone_dirs = []
  for latent_dir in all_latent_dirs:
    if partial_stone_map.latent_pos_dir[latent_dim] == helpers.UNKNOWN:
      latent_dirs_stone_dirs.append((latent_dir, -1))
      latent_dirs_stone_dirs.append((latent_dir, 1))
    else:
      if partial_stone_map.latent_pos_dir[latent_dim] == 1:
        stone_dir = latent_dir
      else:
        stone_dir = -latent_dir
      latent_dirs_stone_dirs.append((latent_dir, stone_dir))
  return latent_dirs_stone_dirs


def reward_plausible(
    latent_direction: int, reward: int,
    plausible_reward_range: Tuple[int, int]
) -> bool:
  """Checks the stone reward is plausible given the latent direction."""
  # If the stone already has the maximum (or minimum) reward it cannot get any
  # higher (or lower).
  if latent_direction * get_num_axes() == reward:
    return False

  # Return if the reward of the new stone is in the plausible range
  lb, ub = plausible_reward_range
  return lb <= reward + (2 * latent_direction) <= ub


def latent_dirs_on_stone(
    perceived_stone: AlignedStone, latent_dim: int,
    partial_stone_map: PartialStoneMap,
    latent_dirs_stone_dirs: Sequence[Tuple[int, int]]
) -> Tuple[bool, List[int]]:
  """Filters possible latent and stone directions given a stone and partial map.

  Args:
    perceived_stone: The stone we are applying a potion to.
    latent_dim: The latent dimension the potion applies to.
    partial_stone_map: Information known about the mapping from stone space to
      latent space
    latent_dirs_stone_dirs: The list of possible latent directions and stone
      directions the potion applies to which we filter.

  Returns:
    Boolean saying if the stone could be unchanged.
    List of still plausible latent directions the stone could move in.
  """
  # If the stone changes it must become the opposite value on the latent
  # dimension passed in.
  expected_stone_dir = -perceived_stone.aligned_coords[latent_dim]
  new_coords = np.copy(perceived_stone.aligned_coords)
  new_coords[latent_dim] = -perceived_stone.aligned_coords[latent_dim]
  # Get the reward for dimensions where we know the latent space coordinates of
  # the stone.
  known_rewards = [c * d for c, d in zip(
      new_coords, partial_stone_map.latent_pos_dir) if d != helpers.UNKNOWN]
  # The reward of the resulting stone could be more or less than the sum of the
  # known rewards by the number of unknown dimensions (extra 1 or -1 per
  # dimension, note that any bonus is applied later).
  unknown_range = len(partial_stone_map.latent_pos_dir) - len(known_rewards)
  known_reward = sum(known_rewards)
  plausible_reward_range = (known_reward - unknown_range,
                            known_reward + unknown_range)

  latent_dirs = [
      latent_dir for latent_dir, stone_dir in latent_dirs_stone_dirs
      if stone_dir == expected_stone_dir and reward_plausible(
          latent_dir, perceived_stone.reward, plausible_reward_range)]

  # If there are 1 or more latent directions that are not possible then the
  # stone would be unchanged in these cases.
  could_stay_still = len(latent_dirs_stone_dirs) > len(latent_dirs)
  return could_stay_still, latent_dirs


class Stone:
  """A stone object in Alchemy."""

  def __init__(self, idx, latent):
    self.idx = idx
    self.latent = np.array(latent)

  def latent_stone(self) -> LatentStone:
    return LatentStone(self.latent)

  def __str__(self) -> str:
    s = 'Stone ' + str(self.idx) + ' (' + str(self.latent) + ')'
    return s

  def __eq__(self, other: 'Stone') -> bool:
    return self.idx == other.idx and all(self.latent == other.latent)

  def __hash__(self) -> int:
    return hash((self.idx, tuple(e for e in self.latent)))

  def __repr__(self) -> str:
    return 'Stone(idx={idx}, latent={latent})'.format(
        idx=self.idx, latent=self.latent)


class Potion:
  """A potion object in Alchemy."""

  def __init__(self, idx, dimension, direction):
    self.idx = idx
    self.dimension = dimension  # 0, 1, or 2
    self.direction = direction  # 1 or -1

  @property
  def as_index(self):
    return self.dimension * 2 + (self.direction + 1) / 2

  def latent_potion(self) -> LatentPotion:
    return LatentPotion(self.dimension, self.direction)

  def __str__(self) -> str:
    return '(' + str(self.dimension) + ',' + str(self.direction) + ')'

  def __eq__(self, other: 'Potion') -> bool:
    return (self.idx == other.idx and self.dimension == other.dimension and
            self.direction == other.direction)

  def __hash__(self) -> int:
    return hash((self.idx, self.dimension, self.direction))

  def same_effect(self, other: 'Potion') -> bool:
    return (self.dimension == other.dimension and
            self.direction == other.direction)

  def __repr__(self) -> str:
    return ('Potion(idx={idx}, dimension={dimension}, '
            'direction={direction})'.format(
                idx=self.idx, dimension=self.dimension,
                direction=self.direction))
