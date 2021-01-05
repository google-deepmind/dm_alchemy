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
"""Trackers running on symbolic alchemy."""

import abc
import collections
import copy
import itertools
from typing import Any, Callable, Dict, Optional, TypeVar

from dm_alchemy import event_tracker
from dm_alchemy.ideal_observer import ideal_observer
from dm_alchemy.ideal_observer import precomputed_maps
from dm_alchemy.types import graphs
from dm_alchemy.types import stones_and_potions
from dm_alchemy.types import utils
import numpy as np

Graph = graphs.Graph

GameState = event_tracker.GameState
NO_OUTCOME = event_tracker.NO_OUTCOME

PerceivedStone = stones_and_potions.PerceivedStone
PerceivedPotion = stones_and_potions.PerceivedPotion
AlignedStoneIndex = stones_and_potions.AlignedStoneIndex
PerceivedPotionIndex = stones_and_potions.PerceivedPotionIndex
StoneMap = stones_and_potions.StoneMap
PotionMap = stones_and_potions.PotionMap
CAULDRON = stones_and_potions.CAULDRON
RewardWeights = stones_and_potions.RewardWeights

PrecomputedMaps = precomputed_maps.PrecomputedMaps

# For typing
symbolic_alchemy = Any


ActionInfo = collections.namedtuple(
    'ActionInfo', 'original_action has_stone has_potion')


# Create a type which can refer to anything derived from SymbolicAlchemyTracker
BaseOrDerivedTracker = TypeVar(
    'BaseOrDerivedTracker', bound='SymbolicAlchemyTracker')


class SequenceStatsTracker:
  """Tracks how a statistic changes throughout an episode."""

  def __init__(
      self, tracker: BaseOrDerivedTracker,
      get_stat: Callable[[BaseOrDerivedTracker], Any],
      default_stat: Any = 0):
    self._get_stat = get_stat
    self._tracker = tracker
    self.stats = []
    self.default_stat = default_stat

  def track(self) -> None:
    self.stats.append(self._get_stat(self._tracker))

  def reset(self) -> None:
    self.stats = []


class SymbolicAlchemyTracker:
  """Object which has functions called for each action in symbolic alchemy."""

  @property
  @abc.abstractmethod
  def name(self) -> str:
    pass

  @property
  def per_action_trackers(self) -> Dict[str, SequenceStatsTracker]:
    return {}

  @property
  def per_trial_trackers(self) -> Dict[str, SequenceStatsTracker]:
    return {}

  def episode_start(self, unused_chemistry: utils.Chemistry) -> None:
    del unused_chemistry
    for tracker in itertools.chain(
        self.per_trial_trackers.values(), self.per_action_trackers.values()):
      tracker.reset()

  def trial_start(self, unused_game_state: GameState) -> None:
    del unused_game_state
    for tracker in self.per_action_trackers.values():
      tracker.track()

  def action_and_outcome(
      self, unused_action: utils.TypeBasedAction,
      unused_outcome: Optional[PerceivedStone], unused_action_info: ActionInfo
  ) -> None:
    del unused_action, unused_outcome, unused_action_info
    for tracker in self.per_action_trackers.values():
      tracker.track()

  def trial_end(self) -> None:
    for tracker in self.per_trial_trackers.values():
      tracker.track()

  def episode_returns(self) -> Any:
    return {k: tuple(tracker.stats)
            for k, tracker in itertools.chain(self.per_trial_trackers.items(),
                                              self.per_action_trackers.items())}

  def default_returns(
      self, num_trials: int, num_actions_per_trial: int
  ) -> Any:
    """Returns some default values for the tracker."""
    per_trial = zip(
        self.per_trial_trackers.items(), itertools.repeat(num_trials))
    num_actions = num_trials * (num_actions_per_trial + 1)
    per_action = zip(
        self.per_action_trackers.items(), itertools.repeat(num_actions))
    return {k: tuple(tracker.default_stat for _ in range(expected_length))
            for (k, tracker), expected_length in itertools.chain(
                per_trial, per_action)}


StatTrackerOrDerived = TypeVar('StatTrackerOrDerived', bound='StatTracker')
GetStat = Callable[[StatTrackerOrDerived, utils.TypeBasedAction,
                    Optional[PerceivedStone], ActionInfo], Any]
Condition = Callable[[utils.TypeBasedAction, Optional[PerceivedStone],
                      ActionInfo], bool]


class StatTracker(SymbolicAlchemyTracker):
  """Tracks a statistic each time an action occurs."""

  def __init__(self, get_stat: GetStat, init_step_stat: Any = 0):
    self._get_stat = get_stat
    self.cumul_action_occurred = copy.deepcopy(init_step_stat)
    self.last_step_stat = copy.deepcopy(init_step_stat)
    self._init_step_stat = init_step_stat
    self.per_action_tracker = SequenceStatsTracker(
        self, lambda tracker: tracker.last_step_stat,
        copy.deepcopy(self._init_step_stat))
    self.per_trial_tracker = SequenceStatsTracker(
        self, lambda tracker: tracker.cumul_action_occurred,
        copy.deepcopy(self._init_step_stat))

  @property
  def per_action_trackers(self) -> Dict[str, SequenceStatsTracker]:
    return {'per_action': self.per_action_tracker}

  @property
  def per_trial_trackers(self) -> Dict[str, SequenceStatsTracker]:
    return {'per_trial': self.per_trial_tracker}

  def action_and_outcome(
      self, action: utils.TypeBasedAction,
      outcome: Optional[PerceivedStone], action_info: ActionInfo
  ) -> None:
    self.last_step_stat = self._get_stat(self, action, outcome, action_info)
    self.cumul_action_occurred += self.last_step_stat
    super().action_and_outcome(action, outcome, action_info)

  def trial_end(self) -> None:
    super().trial_end()
    self.cumul_action_occurred = copy.deepcopy(self._init_step_stat)


class SpecificActionTracker(StatTracker):
  """Counts number of actions which satisfy some condition."""

  def __init__(self, condition: Condition):
    def get_stat(
        unused_tracker: StatTracker, action: utils.TypeBasedAction,
        outcome: Optional[PerceivedStone], action_info: ActionInfo) -> int:
      return 1 if condition(action, outcome, action_info) else 0
    super().__init__(get_stat=get_stat)


class NoChangeActionTracker(SpecificActionTracker):
  """Counts number of actions which do not cause stone to change."""

  NAME = 'no_change'

  @property
  def name(self) -> str:
    return self.NAME

  def __init__(self):
    def condition(
        action: utils.TypeBasedAction, outcome: Optional[PerceivedStone],
        unused_action_info: ActionInfo) -> bool:
      del unused_action_info
      return (all(stone is not None
                  for stone in [outcome, action.perceived_stone]) and
              action.perceived_stone == outcome)
    super().__init__(condition=condition)


class NegStoneCashedTracker(SpecificActionTracker):
  """Counts number of times a negative stone is put in the cauldron."""

  NAME = 'neg_stone'

  @property
  def name(self) -> str:
    return self.NAME

  def __init__(self):
    def condition(
        action: utils.TypeBasedAction, unused_outcome: Optional[PerceivedStone],
        unused_action_info: ActionInfo
    ) -> bool:
      del unused_outcome, unused_action_info
      return (action.cauldron and action.perceived_stone is not None and
              action.perceived_stone.reward < 0)
    super().__init__(condition=condition)


class CashedStoneValueTracker(SymbolicAlchemyTracker):
  """Counts average value of cashed stone."""

  NAME = 'cashed_stone_value'

  @property
  def name(self) -> str:
    return self.NAME

  def __init__(
      self, reward_weights: RewardWeights, stone_map: StoneMap,
      rotation: np.ndarray):
    self._stone_map = stone_map
    self._rotation = rotation
    self.average_stone_value = 0.0
    self._num_stones_cashed = 0
    self._reward_weights = reward_weights
    self.per_trial_tracker = SequenceStatsTracker(
        self, lambda tracker: tracker.average_stone_value, 0.0)

  @property
  def per_trial_trackers(self) -> Dict[str, SequenceStatsTracker]:
    return {'per_trial': self.per_trial_tracker}

  def action_and_outcome(
      self, action: utils.TypeBasedAction, outcome: Optional[PerceivedStone],
      action_info: ActionInfo
  ) -> None:
    if action.cauldron and action.using_stone:
      aligned_stone = stones_and_potions.align(
          action.perceived_stone, self._rotation)
      latent_stone = self._stone_map.apply(aligned_stone)
      self.average_stone_value += self._reward_weights(
          latent_stone.latent_coords)
      self._num_stones_cashed += 1
    super().action_and_outcome(action, outcome, action_info)

  def trial_end(self) -> None:
    if self._num_stones_cashed > 0:
      self.average_stone_value /= self._num_stones_cashed
    super().trial_end()
    self.average_stone_value = 0.0
    self._num_stones_cashed = 0


class ChangeGoldstoneTracker(SpecificActionTracker):
  """Counts number of times a goldstone is changed to something else."""

  NAME = 'gold_changed'

  @property
  def name(self) -> str:
    return self.NAME

  def __init__(self, threshold: int = 2):

    def condition(
        action: utils.TypeBasedAction, outcome: Optional[PerceivedStone],
        unused_action_info: ActionInfo) -> bool:
      del unused_action_info
      if not action.using_stone or not action.using_potion:
        return False
      stone_reward = (action.perceived_stone.reward
                      if action.perceived_stone else 0)
      return outcome is not None and stone_reward > threshold > outcome.reward
    super().__init__(condition=condition)


def pos_stone_not_cashed_tracker_name(
    lb: int = 0, ub: Optional[int] = None
) -> str:
  if lb == 0 and ub is None:
    return 'pos_stone_not_cashed'
  elif ub is None:
    return 'stone_above_' + str(lb) + '_not_cashed'
  return 'stone_between_' + str(lb) + '_and_' + str(ub) + '_not_cashed'


class PosStoneNotCashedTracker(SymbolicAlchemyTracker):
  """Counts number of times a stone with specified reward is not cashed."""

  def __init__(
      self, reward_weights: RewardWeights, lb: int = 0,
      ub: Optional[int] = None):
    self.pos_stones_at_end = 0
    self._condition = lambda r: lb < r < ub if ub is not None else lb < r
    self._game_state = None
    self._reward_weights = reward_weights
    self.lb = lb
    self.ub = ub
    self.per_trial_tracker = SequenceStatsTracker(
        self, lambda tracker: tracker.pos_stones_at_end)

  @property
  def per_trial_trackers(self) -> Dict[str, SequenceStatsTracker]:
    return {'per_trial': self.per_trial_tracker}

  @property
  def name(self) -> str:
    return pos_stone_not_cashed_tracker_name(self.lb, self.ub)

  def trial_start(self, game_state: GameState) -> None:
    self._game_state = game_state
    super().trial_start(game_state)

  def trial_end(self) -> None:
    self.pos_stones_at_end = len(
        [s for s in self._game_state.existing_stones()
         if self._condition(self._reward_weights(s.latent))])
    super().trial_end()


class StoneImprovementTracker(SymbolicAlchemyTracker):
  """Counts number of times a goldstone is changed to something else."""

  # pylint: disable=protected-access
  # TODO(b/173784755): avoid protected access by using event tracker to tracker
  #  latest slot based action.
  NAME = 'stone_improvement'

  @property
  def name(self) -> str:
    return self.NAME

  def __init__(
      self, reward_weights: RewardWeights, stone_map: StoneMap,
      rotation: np.ndarray):
    self._stone_map = stone_map
    self._rotation = rotation
    self.average_stone_improvement = 0.0
    self._reward_weights = reward_weights
    self._game_state = None
    self._start_rewards = {}
    self._end_rewards = {}
    self._prev_existing_stones = set()
    self.per_trial_tracker = SequenceStatsTracker(
        self, lambda tracker: tracker.average_stone_improvement, 0.0)

  @property
  def per_trial_trackers(self) -> Dict[str, SequenceStatsTracker]:
    return {'per_trial': self.per_trial_tracker}

  def action_and_outcome(
      self, action: utils.TypeBasedAction, outcome: Optional[PerceivedStone],
      action_info: ActionInfo
  ) -> None:
    if action.cauldron:
      # We can't get the stone ind as it has already been removed from the game
      # state, so instead just see what stone ind is missing.
      missing_stones = self._prev_existing_stones.difference(
          self._game_state._existing_stones)
      assert len(missing_stones) == 1, (
          'Should be 1 missing stone when stone is used.')
      aligned_stone = stones_and_potions.align(
          action.perceived_stone, self._rotation)
      latent_stone = self._stone_map.apply(aligned_stone)
      for ind in missing_stones:
        self._end_rewards[ind] = self._reward_weights(
            latent_stone.latent_coords)
      self._prev_existing_stones = copy.deepcopy(
          self._game_state._existing_stones)
    super().action_and_outcome(action, outcome, action_info)

  def trial_start(self, game_state: GameState) -> None:
    self._game_state = game_state
    self._prev_existing_stones = copy.deepcopy(
        self._game_state._existing_stones)
    self._start_rewards = {
        i: self._reward_weights(self._game_state.get_stone(i).latent)
        for i in self._prev_existing_stones}
    super().trial_start(game_state)

  def trial_end(self) -> None:
    stone_improvements = [reward - self._start_rewards[idx]
                          for idx, reward in self._end_rewards.items()]
    self.average_stone_improvement = (
        0.0 if not stone_improvements else np.mean(stone_improvements))
    super().trial_end()
    self.average_stone_improvement = 0.0
    self._start_rewards = {}
    self._end_rewards = {}

  # pylint: enable=protected-access


class AddMatrixEventTracker(SymbolicAlchemyTracker):
  """Adds a matrix event tracker per trial and add these to episode returns."""

  NAME = 'matrix_event'

  @property
  def name(self) -> str:
    return self.NAME

  def __init__(self):
    self._event_trackers = None
    self.game_state = None
    self.per_trial_tracker = SequenceStatsTracker(
        self, lambda tracker: tracker.game_state.trackers[self.name],
        event_tracker.MatrixEventTracker(1, 1))

  @property
  def per_trial_trackers(self) -> Dict[str, SequenceStatsTracker]:
    return {'event_tracker': self.per_trial_tracker}

  def trial_start(self, game_state: GameState) -> None:
    matrix_event_tracker = event_tracker.MatrixEventTracker(
        game_state.num_stones, game_state.num_potions)
    self.game_state = game_state
    game_state.add_event_trackers([matrix_event_tracker])
    super().trial_start(game_state)


class ItemGeneratedTracker(SymbolicAlchemyTracker):
  """Tracks the items generated during the episode."""

  NAME = 'items_generated'

  @property
  def name(self) -> str:
    return self.NAME

  def __init__(self):
    self.trials = None
    self.per_trial_tracker = SequenceStatsTracker(
        self, lambda tracker: tracker.trials,
        utils.TrialItems(stones=[], potions=[]))

  @property
  def per_trial_trackers(self) -> Dict[str, SequenceStatsTracker]:
    return {'trials': self.per_trial_tracker}

  def trial_start(self, game_state: GameState) -> None:
    self.trials = copy.deepcopy(game_state.existing_items())
    super().trial_start(game_state)

  def episode_returns(self) -> Any:
    items = utils.EpisodeItems([], [])
    items.trials = super().episode_returns()['trials']
    return items


class ScoreTracker(StatTracker):
  """Adds a reward tracker and return reward per trial."""

  NAME = 'score'

  @property
  def name(self) -> str:
    return self.NAME

  def __init__(self, reward_weights: RewardWeights):
    self._reward_weights = reward_weights
    self.prev_reward = 0
    self.game_state = None

    def latest_reward(tracker, *unused_args, **unused_kwargs):
      del unused_args, unused_kwargs
      cumul_reward = tracker.game_state.trackers['reward'].reward
      reward = cumul_reward - tracker.prev_reward
      tracker.prev_reward = cumul_reward
      return reward

    super().__init__(get_stat=latest_reward)

  def trial_start(self, game_state: GameState) -> None:
    reward_tracker = event_tracker.RewardTracker(self._reward_weights)
    self.game_state = game_state
    game_state.add_event_trackers([reward_tracker])
    self.prev_reward = 0
    super().trial_start(game_state)


class ItemsUsedTracker(StatTracker):
  """Tracks what stones and potions are used."""

  NAME = 'items_used'

  @property
  def name(self) -> str:
    return self.NAME

  def __init__(self):
    self.prev_items = np.zeros((2,), dtype=np.int)
    self.game_state: Optional[GameState] = None

    def latest_items_used(
        tracker: 'ItemsUsedTracker', unused_action: utils.TypeBasedAction,
        unused_outcome: Optional[PerceivedStone], unused_action_info: ActionInfo
    ) -> np.ndarray:
      del unused_action, unused_outcome, unused_action_info
      items_used = tracker.game_state.trackers['items_used']
      cumul_items_used = np.array(
          [items_used.num_potions_used, items_used.num_stones_used],
          dtype=np.int)
      items_used = cumul_items_used - tracker.prev_items
      tracker.prev_items = cumul_items_used
      return items_used

    super().__init__(get_stat=latest_items_used,
                     init_step_stat=np.zeros((2,), dtype=np.int))

  def trial_start(self, game_state: GameState) -> None:
    self.game_state = game_state
    game_state.add_event_trackers([event_tracker.ItemsUsedTracker()])
    self.prev_items = np.zeros((2,), dtype=np.int)
    super().trial_start(game_state)


TrialExtraInfo = collections.namedtuple(
    'TrialExtraInfo',
    'num_world_states num_potion_maps num_stone_maps num_graphs')


class BeliefStateTracker(SymbolicAlchemyTracker):
  """Adds a belief state which is updated to a symbolic alchemy bot."""

  NAME = 'belief_state'

  @property
  def name(self) -> str:
    return self.NAME

  def __init__(
      self, precomputed: PrecomputedMaps,
      env: 'symbolic_alchemy.SymbolicAlchemy',
      init_belief_state=None):
    self.precomputed = precomputed
    self.belief_state = None
    self._init_belief_state = (
        init_belief_state or ideal_observer.BeliefStateWithRotation(
            self.precomputed))
    self._extra_info = None
    self._world_states_per_action = None
    self._env = env
    self.extra_info_per_action_tracker = SequenceStatsTracker(
        self, lambda tracker: tracker.extra_info,
        TrialExtraInfo(
            num_world_states=0, num_stone_maps=0, num_potion_maps=0,
            num_graphs=0))
    self.extra_info_per_trial_tracker = SequenceStatsTracker(
        self, lambda tracker: tracker.extra_info,
        TrialExtraInfo(
            num_world_states=0, num_stone_maps=0, num_potion_maps=0,
            num_graphs=0))

  @property
  def per_action_trackers(self) -> Dict[str, SequenceStatsTracker]:
    return {'per_action_extra_info': self.extra_info_per_action_tracker}

  @property
  def per_trial_trackers(self) -> Dict[str, SequenceStatsTracker]:
    return {'extra_info': self.extra_info_per_trial_tracker}

  def episode_start(self, unused_chemistry: utils.Chemistry):
    self.belief_state = copy.deepcopy(self._init_belief_state)
    super().episode_start(unused_chemistry)

  def trial_start(self, game_state: GameState) -> None:
    current_stones = collections.Counter(self._env.perceived_stones())
    current_potions = collections.Counter(self._env.perceived_potions())
    self.belief_state.new_trial(current_stones, current_potions)
    super().trial_start(game_state)

  def action_and_outcome(
      self, action: utils.TypeBasedAction, outcome: Optional[PerceivedStone],
      action_info: ActionInfo
  ) -> None:
    # A stone value of -1 indicates that the action was invalid
    if not action.using_stone:
      super().action_and_outcome(action, outcome, action_info)
      return
    if action.perceived_stone is None:
      raise ValueError('Action says using stone but perceived stone is None.')
    # An outcome of -1 means the stone did not change.
    current_outcome = outcome or action.perceived_stone
    assert current_outcome is not None
    if action.using_potion:
      self.belief_state.action_and_outcome(
          action.perceived_stone, action.perceived_potion, current_outcome,
          self.precomputed)
    super().action_and_outcome(action, outcome, action_info)

  @property
  def extra_info(self) -> TrialExtraInfo:
    return TrialExtraInfo(
        num_world_states=self.belief_state.num_world_states,
        num_potion_maps=self.belief_state.num_potion_maps,
        num_stone_maps=self.belief_state.num_stone_maps,
        num_graphs=self.belief_state.num_graphs)

  def get_partial_potion_map(
      self, index_to_perm_index: np.ndarray
  ) -> stones_and_potions.PartialPotionMap:
    return self.belief_state.partial_potion_map(index_to_perm_index)

  def get_partial_stone_map(self) -> stones_and_potions.PartialStoneMap:
    return self.belief_state.partial_stone_map()

  def get_partial_graph(
      self, possible_partial_graph_indices: np.ndarray
  ) -> graphs.PartialGraph:
    return self.belief_state.partial_graph(possible_partial_graph_indices)
