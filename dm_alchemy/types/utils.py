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
"""Composite alchemy types."""

import dataclasses
import enum
import math
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Set, Union

from dm_alchemy.ideal_observer import precomputed_maps
from dm_alchemy.types import graphs
from dm_alchemy.types import helpers
from dm_alchemy.types import stones_and_potions
import numpy as np


@dataclasses.dataclass
class Chemistry:
  """The potion map, stone map and graph which together form a chemistry."""
  potion_map: stones_and_potions.PotionMap
  stone_map: stones_and_potions.StoneMap
  graph: graphs.Graph
  rotation: np.ndarray

  def __eq__(self, other: 'Chemistry') -> bool:
    return (self.potion_map == other.potion_map and
            self.stone_map == other.stone_map and
            graphs.constraint_from_graph(self.graph) ==
            graphs.constraint_from_graph(other.graph) and
            stones_and_potions.rotations_equal(self.rotation, other.rotation))


class TrialItems:
  """Stones and potions in a single trial.

  We have 2 different types representing the latent information about stones and
  potions. We accept either and convert so that downstream functions do not have
  to worry about which type they are dealing with.
  """

  def __init__(
      self, potions: Union[Sequence[stones_and_potions.Potion],
                           Sequence[stones_and_potions.LatentPotion]],
      stones: Union[Sequence[stones_and_potions.Stone],
                    Sequence[stones_and_potions.LatentStone]]):
    self.potions = potions
    if potions and isinstance(potions[0], stones_and_potions.LatentPotion):
      latent_potions: Sequence[stones_and_potions.LatentPotion] = potions
      self.potions = [
          stones_and_potions.Potion(i, latent.latent_dim, latent.latent_dir)
          for i, latent in enumerate(latent_potions)]
    self.stones = stones
    if stones and isinstance(stones[0], stones_and_potions.LatentStone):
      latent_stones: Sequence[stones_and_potions.LatentStone] = stones
      self.stones = [
          stones_and_potions.Stone(i, latent.latent_coords)
          for i, latent in enumerate(latent_stones)]

  @property
  def num_stones(self) -> int:
    return len(self.stones)

  @property
  def num_potions(self) -> int:
    return len(self.potions)

  def __repr__(self) -> str:
    return 'TrialItems(potions={potions}, stones={stones})'.format(
        potions=repr(self.potions),
        stones=repr(self.stones))

  def __eq__(self, other: 'TrialItems') -> bool:
    return self.potions == other.potions and self.stones == other.stones


class EpisodeItems:
  """Initial stones and potions for each trial in an episode."""

  def __init__(
      self, potions: Union[Sequence[Sequence[stones_and_potions.Potion]],
                           Sequence[Sequence[stones_and_potions.LatentPotion]]],
      stones: Union[Sequence[Sequence[stones_and_potions.Stone]],
                    Sequence[Sequence[stones_and_potions.LatentStone]]]):
    self.trials = [TrialItems(potions=potions, stones=stones)
                   for potions, stones in zip(potions, stones)]

  @property
  def num_trials(self) -> int:
    return len(self.trials)

  @property
  def stones(self) -> List[List[stones_and_potions.Stone]]:
    return [trial.stones for trial in self.trials]

  @property
  def potions(self) -> List[List[stones_and_potions.Potion]]:
    return [trial.potions for trial in self.trials]

  def __repr__(self) -> str:
    return 'EpisodeItems(trials={trials})'.format(trials=repr(self.trials))

  def __eq__(self, other: 'EpisodeItems') -> bool:
    return self.trials == other.trials


@dataclasses.dataclass
class SymbolicBot:
  run: bool = False
  trackers: Optional[List[str]] = None
  bot_running_trackers: Optional[List[str]] = None


@dataclasses.dataclass
class SymbolicBots:
  """The symbolic bots we can run on alchemy."""
  ideal_observer: SymbolicBot = dataclasses.field(default_factory=SymbolicBot)
  ideal_explorer: SymbolicBot = dataclasses.field(default_factory=SymbolicBot)
  random_action: SymbolicBot = dataclasses.field(default_factory=SymbolicBot)
  search_oracle: SymbolicBot = dataclasses.field(default_factory=SymbolicBot)
  agent_symbolic: SymbolicBot = dataclasses.field(default_factory=SymbolicBot)


class ElementContent(enum.IntEnum):
  """The type of content for each element of a chemistry observation."""
  GROUND_TRUTH = 0
  BELIEF_STATE = 1
  UNKNOWN = 2


class ElementType(enum.IntEnum):
  """The type of content for each element of a chemistry observation."""
  GRAPH = 0
  POTION_MAP = 1
  STONE_MAP = 2
  ROTATION = 3


class GroupInChemistry:
  """A set of dimensions in the chemistry observation."""

  def __init__(
      self, group: Mapping[ElementType, Set[int]], distr: Sequence[float]):
    self.group = group
    self.distr = distr
    if len(self.distr) != len(ElementContent):
      raise ValueError('Must provide a probability for each content type.')
    if abs(sum(self.distr) - 1.0) > 0.0001:
      raise ValueError('Elements of distr must sum to 1.')

  def __repr__(self) -> str:
    return ('GroupInChemistry(group={group}, '
            'distr={distr})'.format(
                group=repr(self.group),
                distr=repr(self.distr)))


GetObsFn = Callable[[], List[float]]


class Element:
  """Details of how we see an element of the chemistry in our observations."""

  def __init__(
      self, element_type: ElementType, expected_obs_length: int,
      present: bool = True):
    """Constructs an element which forms part of the observed chemistry.

    Args:
      element_type: Specifies which part of the chemistry this element refers
        to.
      expected_obs_length: The length of this observation element.
      present: Whether this element is present in the observation.
    """
    self.element_type = element_type
    self.expected_obs_length = expected_obs_length
    self.present = present

  def all_dims(self) -> Set[int]:
    return set(range(self.expected_obs_length))

  def form_observation(
      self, dimensions: Mapping[ElementContent, Set[int]],
      get_content_obs: Mapping[ElementContent, GetObsFn]) -> List[float]:
    """Gets the observation merging different content types.

    Args:
      dimensions: Mapping from different types of content to sets of dimensions
        which hold that content type.
      get_content_obs: Mapping from content types to functions which get the
        observation of that content type.

    Returns:
      The observation with different content type at different dimensions as
      specified.
    """
    if not self.present:
      return []
    # Start with unknown obs
    obs = get_content_obs[ElementContent.UNKNOWN]()
    for content in [ElementContent.GROUND_TRUTH, ElementContent.BELIEF_STATE]:
      # If any of the groups require this type of content then get it and fill
      # in the dimensions for these groups.
      if content in dimensions and dimensions[content]:
        content_obs = get_content_obs[content]()
        for i in dimensions[content]:
          obs[i] = content_obs[i]
    return obs

  def __repr__(self) -> str:
    return ('Element(element_type={element_type}, '
            'expected_obs_length={expected_obs_length}, '
            'present={present})'.format(
                element_type=repr(self.element_type),
                expected_obs_length=repr(self.expected_obs_length),
                present=repr(self.present)))


class PotionMapElement(Element):
  """How we see the potion map element of the chemistry in our observations."""

  def __init__(self, **kwargs):
    super().__init__(
        element_type=ElementType.POTION_MAP, expected_obs_length=9, **kwargs)


class StoneMapElement(Element):
  """How we see the stone map element of the chemistry in our observations."""

  def __init__(self, **kwargs):
    super().__init__(
        element_type=ElementType.STONE_MAP, expected_obs_length=3, **kwargs)


class GraphElement(Element):
  """How we see the graph element of the chemistry in our observations."""

  def __init__(self, **kwargs):
    super().__init__(
        element_type=ElementType.GRAPH, expected_obs_length=12, **kwargs)


class RotationElement(Element):
  """How we see the rotation element of the chemistry in our observations."""

  def __init__(self, **kwargs):
    super().__init__(
        element_type=ElementType.ROTATION, expected_obs_length=4, **kwargs)


@dataclasses.dataclass
class GetChemistryObsFns:
  """Functions to get the observations for different content types for each element."""
  potion_map: Mapping[ElementContent, GetObsFn]
  stone_map: Mapping[ElementContent, GetObsFn]
  graph: Mapping[ElementContent, GetObsFn]
  rotation: Mapping[ElementContent, GetObsFn]

  def element(
      self, element_type: ElementType
  ) -> Mapping[ElementContent, GetObsFn]:
    if element_type == ElementType.POTION_MAP:
      return self.potion_map
    if element_type == ElementType.STONE_MAP:
      return self.stone_map
    if element_type == ElementType.ROTATION:
      return self.rotation
    return self.graph


_EPSILON = 0.0001


class ChemistrySeen:
  """What elements of the chemistry we see in our observations."""

  def __init__(
      self, groups: Optional[Sequence[GroupInChemistry]] = None,
      content: Optional[ElementContent] = None,
      potion_map: Optional[PotionMapElement] = None,
      stone_map: Optional[StoneMapElement] = None,
      graph: Optional[GraphElement] = None,
      rotation: Optional[RotationElement] = None,
      precomputed: Optional[Union[str, precomputed_maps.PrecomputedMaps]] = None
  ):
    """Returns a SeeChemistry object.

    Args:
      groups: Groups of dimensions which get the same content each episode.
      content: The type of content for the chemistry (if all dimensions are in
        the same group).
      potion_map: Information about the potion map element of the chemistry
        observation.
      stone_map: Information about the stone map element of the chemistry
        observation.
      graph: Information about the graph element of the chemistry observation.
      rotation: Information about the rotation element of the chemistry
        observation.
      precomputed: If any of the elements involve computing the belief state
        then precomputed should be either a set of precomputed maps or the level
        name for which we can load the precomputed maps.
    """
    self.potion_map = potion_map or PotionMapElement()
    self.stone_map = stone_map or StoneMapElement()
    self.graph = graph or GraphElement()
    self.rotation = rotation or RotationElement()
    self.precomputed = precomputed
    if groups:
      self.groups = groups
      if content is not None:
        raise ValueError('content ignored if groups is passed in.')
    else:
      # If groups is not passed we set one group for all elements and give all
      # the probability to the content being unknown.
      distr = [0.0 for content in ElementContent]
      if content is None:
        content = ElementContent.UNKNOWN
      distr[content] = 1.0
      self.groups = [GroupInChemistry(
          {element_type: self.element(element_type).all_dims()
           for element_type in ElementType}, distr)]

  def element(self, element_type: ElementType) -> Element:
    """Gets info about the specified element of the chemistry."""
    if element_type == ElementType.POTION_MAP:
      return self.potion_map
    if element_type == ElementType.STONE_MAP:
      return self.stone_map
    if element_type == ElementType.ROTATION:
      return self.rotation
    return self.graph

  def sample_contents(self) -> List[ElementType]:
    """Samples content types for each content group."""
    contents = []
    for group in self.groups:
      contents.append(np.where(np.random.multinomial(1, group.distr))[0][0])
    return contents

  def uses_content_type(self, content: ElementContent) -> bool:
    """Could the chemistry contain the specified content."""
    return any(group.distr[content] > _EPSILON for group in self.groups)

  def dimensions_for_content(
      self, contents: Sequence[ElementContent], element_type: ElementType
  ) -> Dict[ElementContent, Set[int]]:
    """Given sampled group contents constructs map from content type to dimensions."""
    dimensions = {content: set() for content in ElementContent}
    for group_index, group_content in enumerate(contents):
      group = self.groups[group_index].group
      if element_type in group:
        dimensions[group_content] |= group[element_type]
    return dimensions

  def form_observation(
      self, contents: Sequence[ElementContent], get_obs: GetChemistryObsFns
  ) -> List[float]:
    """Forms an observation with the correct content type at each dimension."""
    obs = []
    for element_type in ElementType:
      dimensions = (self.dimensions_for_content(contents, element_type)
                    if contents else {})
      obs.extend(self.element(element_type).form_observation(
          dimensions, get_obs.element(element_type)))
    return obs

  def obs_size(self) -> int:
    """Returns the size of the chemistry observation."""
    # length of observation depends on number of axes
    num_axes = stones_and_potions.get_num_axes()
    vector_size = 0
    if self.stone_map.present:
      # stone pos_dir
      vector_size += num_axes
    if self.potion_map.present:
      # potion dir_map
      vector_size += num_axes
      # potion dim_map
      # TODO(b/173787297): make dim_map factorized; uncomment below
      # vector_size += num_axes * num_axes
      vector_size += math.factorial(num_axes)
    if self.graph.present:
      # edges in a cube of dimension num_axes
      vector_size += graphs.num_edges_in_cube()
    if self.rotation.present:
      vector_size += len(stones_and_potions.possible_rotations())
    return vector_size

  def initialise_precomputed(self) -> None:
    """Loads precomputed maps if necessary."""
    if isinstance(self.precomputed, str):
      # For the purpose of seeing the chemistry precomputed with and without
      # shaping are equivalent and we do not store precomputed for levels
      # with shaping so just use the precomputed for the level without
      precomputed = self.precomputed.replace('_w_shaping', '').replace(
          '_shaping', '')
      self.precomputed = precomputed_maps.load_from_level_name(precomputed)
      if not self.precomputed:
        raise ValueError(
            'Could not load precomputed maps for ' + self.precomputed + '.')

  def __repr__(self) -> str:
    return ('SeeChemistry(groups={groups}, potion_map={potion_map}, '
            'stone_map={stone_map}, graph={graph}, rotation={rotation}, '
            'precomputed={precomputed})'.format(
                groups=repr(self.groups), potion_map=repr(self.potion_map),
                stone_map=repr(self.stone_map), graph=repr(self.graph),
                rotation=repr(self.rotation),
                precomputed=repr(self.precomputed)))


class SlotBasedAction:
  """Represents an action using the stone and potion slot indices."""

  def __init__(
      self, end_trial: bool = False, no_op: bool = False,
      stone_ind: Optional[int] = None, cauldron: bool = False,
      potion_ind: Optional[int] = None):
    self.end_trial = end_trial
    self.no_op = no_op
    self.cauldron = cauldron
    self.stone_ind = stone_ind
    self.potion_ind = potion_ind

  def _valid(self) -> bool:
    """Action is valid if exactly one of these things is true."""
    put_stone_in_cauldron = self.cauldron and self.using_stone
    put_stone_in_potion = self.using_potion and self.using_stone
    return [self.end_trial, self.no_op, put_stone_in_cauldron,
            put_stone_in_potion].count(True) == 1

  @property
  def using_stone(self) -> bool:
    return self.stone_ind is not None

  @property
  def using_potion(self) -> bool:
    return self.potion_ind is not None

  def __repr__(self):
    return (
        'SlotBasedAction(end_trial={end_trial}, no_op={no_op}, '
        'stone_ind={stone_ind}, cauldron={cauldron}, potion_ind={potion_ind})'.
        format(end_trial=self.end_trial, no_op=self.no_op,
               stone_ind=self.stone_ind, cauldron=self.cauldron,
               potion_ind=self.potion_ind))


class TypeBasedAction:
  """Represents an action using the stone and potion types."""

  def __init__(
      self, end_trial: bool = False, no_op: bool = False,
      stone: Optional[stones_and_potions.PerceivedStone] = None,
      cauldron: bool = False,
      potion: Optional[stones_and_potions.PerceivedPotion] = None):
    self.end_trial = end_trial
    self.no_op = no_op
    self.cauldron = cauldron
    self.perceived_stone = stone
    self.perceived_potion = potion

  def _valid(self) -> bool:
    """Action is valid if exactly one of these things is true."""
    put_stone_in_cauldron = self.cauldron and self.using_stone
    put_stone_in_potion = self.using_potion and self.using_stone
    return [self.end_trial, self.no_op, put_stone_in_cauldron,
            put_stone_in_potion].count(True) == 1

  @property
  def using_stone(self) -> bool:
    return self.perceived_stone is not None

  @property
  def using_potion(self) -> bool:
    return self.perceived_potion is not None

  def __repr__(self):
    return (
        'TypeBasedAction(end_trial={end_trial}, no_op={no_op}, stone={stone}, '
        'cauldron={cauldron}, potion={potion})'.format(
            end_trial=self.end_trial, no_op=self.no_op,
            stone=self.perceived_stone, cauldron=self.cauldron,
            potion=self.perceived_potion))


def type_based_action_from_ints(
    aligned_stone_index: stones_and_potions.AlignedStoneIndex,
    perceived_potion_index: stones_and_potions.PerceivedPotionIndex,
    rotation: np.ndarray
) -> TypeBasedAction:
  """Converts from int specification of action to type based."""
  if aligned_stone_index == helpers.END_TRIAL:
    return TypeBasedAction(end_trial=True)
  perceived_stone = stones_and_potions.unalign(
      stones_and_potions.aligned_stone_from_index(aligned_stone_index),
      rotation)
  if perceived_potion_index == stones_and_potions.CAULDRON:
    return TypeBasedAction(stone=perceived_stone, cauldron=True)
  perceived_potion = stones_and_potions.perceived_potion_from_index(
      perceived_potion_index)
  return TypeBasedAction(stone=perceived_stone, potion=perceived_potion)
