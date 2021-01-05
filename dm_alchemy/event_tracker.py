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
"""Tracks the order of alchemy events and resulting stones and potions."""

import abc
import collections
import copy
import itertools
import random
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

from dm_alchemy.types import graphs
from dm_alchemy.types import stones_and_potions
from dm_alchemy.types import utils
import numpy as np


Stone = stones_and_potions.Stone
Potion = stones_and_potions.Potion
LatentStone = stones_and_potions.LatentStone
LatentPotion = stones_and_potions.LatentPotion
AlignedStone = stones_and_potions.AlignedStone
PerceivedPotion = stones_and_potions.PerceivedPotion
StoneMap = stones_and_potions.StoneMap
PotionMap = stones_and_potions.PotionMap
CAULDRON = stones_and_potions.CAULDRON
RewardWeights = stones_and_potions.RewardWeights
Graph = graphs.Graph

NEVER_USED = -1
NO_OUTCOME = -1
UNKNOWN_TYPE = -3


class EventTracker(abc.ABC):
  """Base class for things that track alchemy events."""

  def __init__(self, name):
    self.name = name

  @abc.abstractmethod
  def potion_used(
      self, stone_ind: int, potion_ind: int, val: int, start_stone: graphs.Node,
      stone_inst: int, potion: Potion, end_stone: graphs.Node) -> None:
    pass

  def failed_potion_use(
      self, stone_ind: int, start_stone: graphs.Node, stone_inst: int) -> None:
    """Optional callback when a potion use is attempted but fails."""
    pass


class GameState:
  """Keeps track of the symbolic state of an alchemy game."""

  def __init__(
      self, graph: graphs.Graph, trial_items: utils.TrialItems,
      event_trackers: Optional[Sequence[EventTracker]] = None
  ):

    self._stones = copy.deepcopy(trial_items.stones)
    self._stone_idx_to_ind = {p.idx: i for i, p in enumerate(self._stones)}
    self._stone_ind_to_idx = {i: p.idx for i, p in enumerate(self._stones)}
    self._potions = copy.deepcopy(trial_items.potions)
    self._potion_idx_to_ind = {p.idx: i for i, p in enumerate(self._potions)}

    self._graph = graph
    num_stones = len(self._stones)
    num_potions = len(self._potions)
    self._existing_stones = set(range(num_stones))
    self._existing_potions = set(range(num_potions))
    trackers = event_trackers if event_trackers is not None else []
    self.trackers = {tracker.name: tracker for tracker in trackers}
    self._count = 0

  def add_event_trackers(self, event_trackers: Sequence[EventTracker]) -> None:
    """Adds event trackers if they are not already there."""
    self.trackers.update({tracker.name: tracker for tracker in event_trackers})

  def get_stone_ind(
      self, stone_inst: Optional[int] = None,
      stone: Optional[Union[graphs.Node, LatentStone]] = None
  ) -> Optional[int]:
    """Gets a stone referred to through a variety of methods.

    The caller must pass exactly one of stone_inst and stone.

    Args:
      stone_inst: The instance id of the stone used in the potion.
      stone: The stone used.

    Returns:
      The index (into the list of stones originally passed to the EventTracker
      in construction) for the stone used in the potion or None if no match can
      be found.
    """
    if len([e for e in [stone_inst, stone] if e is not None]) != 1:
      raise ValueError('Exactly one of stone inst and stone must be given.')

    if stone_inst is not None:
      return self._stone_idx_to_ind[stone_inst]

    if isinstance(stone, LatentStone):
      stone_node = graphs.Node(-1, stone.latent_coords)
    else:
      stone_node = stone
    matches = self._matching_stones(stone_node)
    if not matches:
      return None
    return matches[0]

  def get_potion_ind(
      self, potion_inst: Optional[int] = None,
      potion: Optional[Union[Potion, LatentPotion]] = None) -> Optional[int]:
    """Gets a potion referred to through a variety of methods.

    The caller must pass exactly one of potion_inst and potion.

    Args:
      potion_inst: The instance id of the potion used.
      potion: The potion used.

    Returns:
      The index (into the list of potions originally passed to the EventTracker
      in construction) for the potion used or None if no match can be found.
      -1 refers to the cauldron.
    """
    if len([e for e in [potion_inst, potion] if e is not None]) != 1:
      raise ValueError('Exactly one of potion inst and potion must be given.')

    if potion_inst is not None:
      return self._potion_idx_to_ind[potion_inst]

    if isinstance(potion, LatentPotion):
      potion = Potion(-1, potion.latent_dim, potion.latent_dir)

    matches = self._matching_potions(potion)
    if not matches:
      return None
    return matches[0]

  def _stone_node(self, ind: int) -> graphs.Node:
    node_ = self._graph.node_list.get_node_by_coords(
        list(self._stones[ind].latent))
    assert node_ is not None
    node: graphs.Node = node_
    return node

  def _matching_potions(self, potion: Potion) -> List[int]:
    return [p for p in self._existing_potions
            if self._potions[p].as_index == potion.as_index]

  def _matching_stones(self, stone_node: graphs.Node) -> List[int]:
    return [i for i in self._existing_stones
            if tuple(self._stone_node(i).coords) == tuple(stone_node.coords)]

  def has_stone_ind(self, stone_ind: int) -> bool:
    return stone_ind in self._existing_stones

  def has_potion_ind(self, potion_ind: int) -> bool:
    return potion_ind in self._existing_potions

  def _remove_potion(self, potion_ind: int) -> None:
    self._existing_potions.remove(potion_ind)

  def _remove_stone(self, stone_ind: int) -> None:
    self._existing_stones.remove(stone_ind)

  def potion_used(
      self, stone_ind: int, potion_ind: int,
      val: Optional[int] = None
  ) -> int:
    """Records that a potion has been used.

    The caller must pass exactly one of stone_ind, stone_inst and stone, and
    exactly one of potion_ind, potion_inst and potion.

    Args:
      stone_ind: The index (into the list of stones originally passed to the
        EventTracker in construction) for the stone used in the potion.
      potion_ind: The index (into the list of potions originally passed to the
        EventTracker in construction) for the potion used. -1 refers to the
        cauldron.
      val: The value to record in this event (typically the frame number that
        this event occurs). If this is not set then the value set will be
        arbitrary but will preserve the order in which the potion_used and
        stone_used functions are called.

    Returns:
      The index (into the list of stones originally passed to the EventTracker
      in construction) for the stone used in the potion. This may not have been
      passed into the function (if stone_inst or stone was passed instead).
    """
    # -1 corresponds to the cauldron and so there is no potion to remove and the
    # stone does not change
    old_node = self._stone_node(stone_ind)
    outcome_stone = None
    potion = None
    if potion_ind != CAULDRON:
      outcome_stone = copy.deepcopy(old_node)
      potion = self._potions[potion_ind]
      # Change the stone in _stones
      if old_node in self._graph.edge_list.edges:
        outcome_stone = [end_node for end_node, v in
                         self._graph.edge_list.edges[old_node].items()
                         if potion.same_effect(v[1])]
        if outcome_stone:
          assert len(outcome_stone) == 1
          outcome_stone = outcome_stone[0]
          self._stones[stone_ind].latent = np.array(list(outcome_stone.coords))
        else:
          outcome_stone = old_node

      self._remove_potion(potion_ind)

    if self.trackers:
      if val is None:
        val = self._count
        self._count += 1
      for event_tracker in self.trackers.values():
        event_tracker.potion_used(
            stone_ind, potion_ind, val, old_node,
            self._stone_ind_to_idx[stone_ind], potion, outcome_stone)

    return stone_ind

  def stone_used(self, stone_ind: int, val: Optional[int] = None) -> None:
    """Records that a stone has been used (placed in the cauldron).

    The caller must pass exactly one of stone_ind, stone_inst and stone.

    Args:
      stone_ind: The index (into the list of stones originally passed to the
        EventTracker in construction) for the stone used in the potion.
      val: The value to record in this event (typically the frame number that
        this event occurs). If this is not set then the value set will be
        arbitrary but will preserve the order in which the potion_used and
        stone_used functions are called.
    """
    self.potion_used(
        stone_ind=stone_ind, potion_ind=CAULDRON, val=val)
    self._remove_stone(stone_ind)

  def failed_potion_use(self, stone_ind: int) -> None:
    old_node = self._stone_node(stone_ind)
    for event_tracker in self.trackers.values():
      event_tracker.failed_potion_use(
          stone_ind, old_node, self._stone_ind_to_idx[stone_ind])

  def has_stones(self) -> bool:
    return bool(self._existing_stones)

  def has_potions(self) -> bool:
    return bool(self._existing_potions)

  def has_stones_and_potions(self) -> bool:
    return self.has_stones() and self.has_potions()

  def rand_stone_ind(self) -> int:
    return random.sample(self._existing_stones, 1)[0]

  def rand_potion_ind(self) -> int:
    return random.sample(self._existing_potions, 1)[0]

  def use_rand_stone_potion_pair(self) -> Tuple[Stone, int]:
    """Uses a random stone with a random potion.

    Returns:
       The new value of the stone and the index of that stone.
    """
    stone_index = self.rand_stone_ind()
    return self.use_rand_potion(stone_index)

  def use_rand_potion(self, stone_ind: int) -> Tuple[Stone, int]:
    """Uses the stone passed with a random potion.

    Args:
      stone_ind: The index (into the list of stones originally passed to the
        EventTracker in construction) for the stone to use in a random potion.

    Returns:
       The new value of the stone and the index of that stone.
    """
    potion_index = self.rand_potion_ind()
    self.potion_used(stone_ind, potion_index)
    return self._stones[stone_ind], stone_ind

  def existing_stone_nodes(self) -> List[graphs.Node]:
    """Returns a list of nodes for the remaining existing stones."""
    return [self._stone_node(i) for i in self._existing_stones]

  def existing_stones(self) -> List[Stone]:
    """Returns a list of the remaining existing stones."""
    return [self._stones[i] for i in self._existing_stones]

  def existing_potions(self) -> List[Potion]:
    """Returns a list of the remaining existing potions."""
    return [self._potions[i] for i in self._existing_potions]

  def existing_items(self) -> utils.TrialItems:
    return utils.TrialItems(
        stones=self.existing_stones(), potions=self.existing_potions())

  @property
  def num_stones(self) -> int:
    return len(self._existing_stones)

  @property
  def num_potions(self) -> int:
    return len(self._existing_potions)

  def check_have_potions(self, needed_potions: Sequence[Potion]) -> bool:
    """Checks that we have all the potions we need."""
    need = collections.Counter([p.as_index for p in needed_potions])
    have = collections.Counter([self._potions[p].as_index
                                for p in self._existing_potions])
    for k in need.keys():
      if k not in have.keys():
        return False
      else:
        if have[k] < need[k]:
          return False
    return True

  def get_stones_above_thresh(
      self, reward_weights: RewardWeights, threshold: int) -> List[int]:
    """Gets all the stones whose value exceeds the threshold passed in."""
    current_vals = {i: reward_weights(self._stones[i].latent)
                    for i in self._existing_stones}
    return [i for i, current_val in current_vals.items()
            if current_val > threshold]

  def use_stones_above_thresh(
      self, reward_weights: RewardWeights, threshold: int) -> None:
    """Uses all the stones whose value exceeds the threshold passed in."""
    for i in self.get_stones_above_thresh(reward_weights, threshold):
      self.stone_used(i)

  def get_stone(self, ind: int) -> Stone:
    return self._stones[ind]

  def get_potion(self, ind: int) -> Potion:
    return self._potions[ind]

  @property
  def node_list(self) -> graphs.NodeList:
    return self._graph.node_list

  @property
  def edge_list(self) -> graphs.EdgeList:
    return self._graph.edge_list

  @property
  def stone_ind_to_idx(self) -> Dict[int, int]:
    return self._stone_ind_to_idx

  @property
  def stone_idx_to_ind(self) -> Dict[int, int]:
    return self._stone_idx_to_ind

  @property
  def potion_idx_to_ind(self) -> Dict[int, int]:
    return self._potion_idx_to_ind


class TrialTracker(EventTracker):
  """Type which tracks all events in a trial."""

  @abc.abstractmethod
  def events_list(self) -> List[Tuple[int, int, int]]:
    """Returns a list of stone index, potion index, val for the trial events."""
    pass


class MatrixEventTracker(TrialTracker):
  """Tracks the order of potion used and stone used events in matrix."""

  def __init__(self, num_stones: int, num_potions: int):
    self.events = np.full(
        shape=(num_stones, num_potions + 1), fill_value=-1, dtype=np.int)
    super().__init__(name='matrix_event')

  def potion_used(
      self, stone_ind: int, potion_ind: int, val: int,
      start_stone: graphs.Node, stone_inst: int, potion: Potion,
      end_stone: graphs.Node) -> None:
    """Records that a potion has been used.

    Args:
      stone_ind: The index (into the list of stones originally passed to the
        EventTracker in construction) for the stone used in the potion.
      potion_ind: The index (into the list of potions originally passed to the
        EventTracker in construction) for the potion used. -1 refers to the
        cauldron.
      val: The value to record in this event (typically the frame number that
        this event occurs). If this is not set then the value set will be
        arbitrary but will preserve the order in which the potion_used and
        stone_used functions are called.
      start_stone: The stone node before the potion is used.
      stone_inst: The instance id for the stone we are using.
      potion: The potion used.
      end_stone: The stone node after the potion is used.
    """
    self.events[stone_ind, potion_ind] = val

  def events_list(self) -> List[Tuple[int, int, int]]:
    stone_used, potion_used = np.where(self.events != -1)
    frame = [self.events[x, y] for (x, y) in zip(stone_used, potion_used)]
    num_potions = self.events.shape[1] - 1
    events = sorted(zip(stone_used, potion_used, frame), key=lambda x: x[2])
    return [
        (stone_ind, CAULDRON if potion_ind == num_potions else potion_ind,
         frame) for stone_ind, potion_ind, frame in events]


ActionSequenceElement = Tuple[int, Mapping[str, Any], int, int]


class ActionSequenceTracker(TrialTracker):
  """Tracks the order of potion used and stone used events in matrix."""

  def __init__(self):
    self._action_sequence = []
    super().__init__(name='action_sequence')

  def potion_used(
      self, stone_ind: int, potion_ind: int, val: int,
      start_stone: graphs.Node, stone_inst: int, potion: Potion,
      end_stone: graphs.Node) -> None:
    """Records that a potion has been used.

    Args:
      stone_ind: The index (into the list of stones originally passed to the
        EventTracker in construction) for the stone used in the potion.
      potion_ind: The index (into the list of potions originally passed to the
        EventTracker in construction) for the potion used. -1 refers to the
        cauldron.
      val: The value to record in this event (typically the frame number that
        this event occurs). If this is not set then the value set will be
        arbitrary but will preserve the order in which the potion_used and
        stone_used functions are called.
      start_stone: The stone node before the potion is used.
      stone_inst: The instance id for the stone we are using.
      potion: The potion used.
      end_stone: The stone node after the potion is used.
    """
    # add to action sequence
    action_dict = {'node': (start_stone.idx, start_stone.coords),
                   'stone_idx': stone_inst}

    # -1 corresponds to the cauldron and so there is no potion to remove and the
    # stone does not change
    if potion_ind == CAULDRON:
      action_dict['action'] = 'cauldron'
    else:
      # Change the stone in _stones
      action_dict['action'] = (potion.as_index,
                               (potion.dimension, potion.direction))
      action_dict['potion_idx'] = potion.idx
      action_dict['outcome_node'] = (end_stone.idx, end_stone.coords)

    self._action_sequence.append((val, action_dict, stone_ind, potion_ind))

  @property
  def action_sequence(self) -> List[Tuple[int, Dict[str, Any], int, int]]:
    self._action_sequence.sort(key=lambda x: x[0])
    return self._action_sequence

  def events_list(self) -> List[Tuple[int, int, int]]:
    return [(stone_ind, potion_ind, val)
            for val, _, stone_ind, potion_ind in self.action_sequence]


class LatestOutcomeTracker(EventTracker):
  """Tracks the most recent outcome of using a potion."""

  def __init__(
      self, potion_map: PotionMap, stone_map: StoneMap, rotation: np.ndarray):
    # -1 represents no change and is the default value for outcome.
    self.outcome = None
    self.type_based_action = None
    self._potion_map, self._stone_map = potion_map, stone_map
    self._rotation = rotation
    super().__init__(name='latest_outcome')

  def reset(self) -> None:
    self.outcome = None
    self.type_based_action = None

  def _perceived_stone(self, stone: graphs.Node):
    aligned_stone = self._stone_map.apply_inverse(LatentStone(np.array(
        stone.coords)))
    return stones_and_potions.unalign(aligned_stone, self._rotation)

  def potion_used(
      self, stone_ind: int, potion_ind: int, val: int,
      start_stone: graphs.Node, stone_inst: int, potion: Potion,
      end_stone: Optional[graphs.Node]) -> None:
    if end_stone is not None:
      aligned_stone = self._stone_map.apply_inverse(LatentStone(np.array(
          end_stone.coords)))
      self.outcome = stones_and_potions.unalign(aligned_stone, self._rotation)
    perceived_stone = self._perceived_stone(start_stone)
    if potion_ind == CAULDRON:
      self.type_based_action = utils.TypeBasedAction(
          stone=perceived_stone, cauldron=True)
    else:
      perceived_potion = self._potion_map.apply_inverse(LatentPotion(
          potion.dimension, potion.direction))
      self.type_based_action = utils.TypeBasedAction(
          stone=perceived_stone, potion=perceived_potion)

  def failed_potion_use(
      self, stone_ind: int, start_stone: graphs.Node, stone_inst: int):
    """Optional callback when a potion use is attempted but fails."""
    self.outcome = None
    perceived_stone = self._perceived_stone(start_stone)
    # This is an invalid action but the stone type can be used for
    # visualization.
    self.type_based_action = utils.TypeBasedAction(stone=perceived_stone)


class RewardTracker(EventTracker):
  """Tracks the reward obtained."""

  def __init__(self, reward_weights: RewardWeights):
    self._reward = 0
    self._reward_weights = reward_weights
    super().__init__(name='reward')

  def potion_used(
      self, stone_ind: int, potion_ind: int, val: int,
      start_stone: graphs.Node, stone_inst: int, potion: Potion,
      end_stone: graphs.Node) -> None:
    """Adds reward when a potion has been used.

    Args:
      stone_ind: The index (into the list of stones originally passed to the
        EventTracker in construction) for the stone used in the potion.
      potion_ind: The index (into the list of potions originally passed to the
        EventTracker in construction) for the potion used. -1 refers to the
        cauldron.
      val: The value to record in this event (typically the frame number that
        this event occurs). If this is not set then the value set will be
        arbitrary but will preserve the order in which the potion_used and
        stone_used functions are called.
      start_stone: The stone node before the potion is used.
      stone_inst: The instance id for the stone we are using.
      potion: The potion used.
      end_stone: The stone node after the potion is used.
    """
    if potion_ind == CAULDRON:
      self._reward += self._reward_weights(start_stone.coords)

  @property
  def reward(self) -> int:
    return self._reward


class ItemsUsedTracker(EventTracker):
  """Tracks the stones and potions used."""

  def __init__(self):
    self.potions_used = []
    self.stones_used = []
    super().__init__(name='items_used')

  def potion_used(
      self, stone_ind: int, potion_ind: int, val: int,
      start_stone: graphs.Node, stone_inst: int, potion: Potion,
      end_stone: graphs.Node) -> None:
    """Keeps lists of potions and stones which have been used.

    Args:
      stone_ind: The index (into the list of stones originally passed to the
        EventTracker in construction) for the stone used in the potion.
      potion_ind: The index (into the list of potions originally passed to the
        EventTracker in construction) for the potion used. -1 refers to the
        cauldron.
      val: The value to record in this event (typically the frame number that
        this event occurs). This is not relevant for this tracker.
      start_stone: The stone node before the potion is used.
      stone_inst: The instance id for the stone we are using.
      potion: The potion used.
      end_stone: The stone node after the potion is used.
    """
    if potion_ind == CAULDRON:
      self.stones_used.append(stone_ind)
    else:
      self.potions_used.append(potion_ind)

  @property
  def num_potions_used(self) -> int:
    return len(self.potions_used)

  @property
  def num_stones_used(self) -> int:
    return len(self.stones_used)


class Event(abc.ABC):
  """Abstract base class for events we want to check in the event tracker."""

  @abc.abstractmethod
  def next_occurrence(
      self, events: np.ndarray) -> Tuple[int, int, Optional[Set[int]]]:
    pass

  def occurs(self, events: np.ndarray) -> bool:
    event_start, _, _ = self.next_occurrence(events)
    not_occurred = event_start == NEVER_USED
    return not not_occurred


class SingleEvent(Event):
  """A single event where a stone is used with one of a set of potions."""

  def __init__(self, stone_ind: int, potion_inds: Set[int]):
    self.stone_ind = stone_ind
    self.potion_inds = potion_inds

  def next_occurrence(
      self, events: np.ndarray) -> Tuple[int, int, Optional[Set[int]]]:
    """Gets the next occurrence of this event.

    Args:
      events: numpy array of stones against potions with the last entry
        corresponding to the cauldron with a -1 in places where that stone was
        never used with that potion and the time of usage otherwise.

    Returns:
      When event starts, when event ends, which potions were used by event.
    """
    frames_potions = [(events[self.stone_ind, p], p) for p in self.potion_inds
                      if events[self.stone_ind, p] >= 0]
    if not frames_potions:
      return NEVER_USED, NEVER_USED, None
    frame, potion_used = min(frames_potions, key=lambda v: v[0])
    return frame, frame, {potion_used}


class AnyOrderEvents(Event):
  """A set of events which can happen in any order."""

  def __init__(self, set_events: Set[Event]):
    self.set_events = set_events

  def next_occurrence(
      self, events: np.ndarray) -> Tuple[int, int, Optional[Set[int]]]:
    """Gets the next occurrence of this event.

    Args:
      events: numpy array of stones against potions with the last entry
        corresponding to the cauldron with a -1 in places where that stone was
        never used with that potion and the time of usage otherwise.

    Returns:
      When event starts, when event ends, which potions were used by event.
    """
    results = [e.next_occurrence(events) for e in self.set_events]
    if any(v[0] == NEVER_USED for v in results):
      return NEVER_USED, NEVER_USED, None
    return (min(v[0] for v in results), max(v[1] for v in results),
            set(itertools.chain.from_iterable([v[2] for v in results])))


class OrderedEvents(Event):
  """A list of events which must happen in the order passed in."""

  def __init__(self, iter_events: Sequence[Event]):
    self.iter_events = iter_events

  def next_occurrence(
      self, events: np.ndarray) -> Tuple[int, int, Optional[Set[int]]]:
    """Gets the next occurrence of this event.

    Args:
      events: numpy array of stones against potions with the last entry
        corresponding to the cauldron with a -1 in places where that stone was
        never used with that potion and the time of usage otherwise.

    Returns:
      When event starts, when event ends, which potions were used by event.
    """
    results = [e.next_occurrence(events) for e in self.iter_events]
    if any(v[0] == NEVER_USED for v in results):
      return NEVER_USED, NEVER_USED, None
    for end_first, start_next in zip([v[1] for v in results[:-1]],
                                     [v[0] for v in results[1:]]):
      # If the events happen on the same step this is allowed.
      if end_first > start_next:
        return NEVER_USED, NEVER_USED, None
    return (results[0][0], results[-1][1],
            set(itertools.chain.from_iterable([v[2] for v in results])))


def replay_events(game_state: GameState, event_tracker: TrialTracker) -> None:
  for stone_ind, potion_ind, val in event_tracker.events_list():
    if potion_ind == CAULDRON:
      game_state.stone_used(stone_ind=stone_ind, val=val)
    else:
      game_state.potion_used(
          stone_ind=stone_ind, potion_ind=potion_ind, val=val)


def matrix_events_to_action_sequence(
    adj_mat: np.ndarray, items: utils.TrialItems,
    matrix_events: MatrixEventTracker
) -> List[ActionSequenceElement]:
  """Takes events/output of evaluation analysis and creates an event tracker."""
  graph = graphs.convert_adj_mat_to_graph(adj_mat)
  action_sequence_tracker = ActionSequenceTracker()
  game_state = GameState(
      trial_items=items, graph=graph, event_trackers=[action_sequence_tracker])
  if matrix_events.events.shape != (items.num_stones, items.num_potions + 1):
    raise ValueError(
        'Matrix of events shape does not match the number of stones and '
        'potions present.')

  replay_events(game_state, matrix_events)
  return action_sequence_tracker.action_sequence
