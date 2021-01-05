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
"""Bots which run on symbolic environment for alchemy."""

import abc
import random
from typing import Any, Callable, Dict, Sequence, Union

from dm_alchemy import event_tracker
from dm_alchemy import symbolic_alchemy
from dm_alchemy import symbolic_alchemy_trackers
from dm_alchemy.ideal_observer import ideal_observer
from dm_alchemy.ideal_observer import precomputed_maps
from dm_alchemy.types import stones_and_potions
from dm_alchemy.types import utils

LatentStone = stones_and_potions.LatentStone
LatentPotion = stones_and_potions.LatentPotion
PotionMap = stones_and_potions.PotionMap
StoneMap = stones_and_potions.StoneMap
AlignedStone = stones_and_potions.AlignedStone
PerceivedPotion = stones_and_potions.PerceivedPotion
RewardWeights = stones_and_potions.RewardWeights

PrecomputedMaps = precomputed_maps.PrecomputedMaps

AddMatrixEventTracker = symbolic_alchemy_trackers.AddMatrixEventTracker
BeliefStateTracker = symbolic_alchemy_trackers.BeliefStateTracker
ScoreTracker = symbolic_alchemy_trackers.ScoreTracker


class SymbolicAlchemyBot(abc.ABC):
  """Bot running on the symbolic alchemy environment."""

  def __init__(self, env: symbolic_alchemy.SymbolicAlchemy):
    self._env = env

  @abc.abstractmethod
  def select_action(
      self
  ) -> Union[utils.SlotBasedAction, utils.TypeBasedAction]:
    pass

  def run_episode(self) -> Dict[str, Any]:
    """Runs the bot on an episode of the symbolic alchemy env."""
    timestep = self._env.reset()
    while not timestep.last():
      action = self.select_action()
      timestep = symbolic_alchemy.take_simplified_action(action, self._env)

    return self._env.episode_returns()


class IdealObserverBot(SymbolicAlchemyBot):
  """Bot which runs the ideal observer on the symbolic alchemy environment."""

  def __init__(
      self, reward_weights: RewardWeights, precomputed: PrecomputedMaps,
      env: symbolic_alchemy.SymbolicAlchemy,
      minimise_world_states: bool):
    self._bonus = reward_weights.bonus
    self._precomputed = precomputed
    self._minimise_world_states = minimise_world_states
    super().__init__(env)
    self._search_results = None

  def run_episode(self) -> Dict[str, Any]:
    # Start with no search results.
    self._search_results = {}
    return super().run_episode()

  def select_action(self) -> utils.TypeBasedAction:
    belief_state_tracker: BeliefStateTracker = (
        self._env.trackers['belief_state'])
    action, _, self._search_results = ideal_observer.ideal_observer(
        belief_state_tracker.belief_state.belief_state, self._search_results,
        self._bonus, self._precomputed, self._minimise_world_states)
    return utils.type_based_action_from_ints(
        *action, belief_state_tracker.belief_state.rotation)


class RandomActionBot(SymbolicAlchemyBot):
  """Bot which takes random actions on the symbolic alchemy environment.

  If a stone reaches the maximum value it will not change it further. When there
  are no more potions or all of the stones have reached the maximum possible
  value then the positive stones are put into the cauldron.
  """

  def __init__(
      self, reward_weights: RewardWeights,
      env: symbolic_alchemy.SymbolicAlchemy, threshold_for_leaving: int = 2
  ):
    self._reward_weights = reward_weights
    self._threshold_for_leaving = threshold_for_leaving
    super().__init__(env)

  def select_action(self) -> utils.TypeBasedAction:
    # Get stones which are not the maximum
    stones = [s for s in  self._env.game_state.existing_stones()
              if self._reward_weights(s.latent) < self._threshold_for_leaving]
    potions = self._env.game_state.existing_potions()
    stones_to_use = [self._env.perceived_stone(s) for s in stones]
    potions_to_use = [self._env.perceived_potion(p) for p in potions]

    if not stones_to_use or not potions_to_use:
      return utils.TypeBasedAction(end_trial=True)
    return utils.TypeBasedAction(
        stone=random.sample(stones_to_use, 1)[0],
        potion=random.sample(potions_to_use, 1)[0])


class ReplayBot(SymbolicAlchemyBot):
  """Bot which replays a sequence of actions."""

  def __init__(
      self, trial_trackers: Sequence[event_tracker.TrialTracker],
      env: symbolic_alchemy.SymbolicAlchemy):
    self._actions = []
    for trial_tracker in trial_trackers:
      self._actions.append([
          (stone_ind, potion_ind)
          for stone_ind, potion_ind, _ in trial_tracker.events_list()])
    self._action_num = [0 for _ in trial_trackers]
    super().__init__(env)

  def select_action(self) -> utils.SlotBasedAction:
    if 0 <= self._env.trial_number < len(self._actions):
      action_num = self._action_num[self._env.trial_number]
      actions = self._actions[self._env.trial_number]
      if action_num < len(actions):
        action = actions[action_num]
        self._action_num[self._env.trial_number] += 1
        stone_ind, potion_ind = action
        if potion_ind == -1:
          return utils.SlotBasedAction(stone_ind=stone_ind, cauldron=True)
        return utils.SlotBasedAction(stone_ind=stone_ind, potion_ind=potion_ind)
    return utils.SlotBasedAction(no_op=True)


class NoOpBot(SymbolicAlchemyBot):
  """Bot which always selects no op actions."""

  def select_action(self) -> utils.SlotBasedAction:
    return utils.SlotBasedAction(no_op=True)


def run_symbolic_alchemy_bot(
    episode_items: utils.EpisodeItems, chemistry: utils.Chemistry,
    reward_weights: RewardWeights,
    bot_from_env: Callable[[symbolic_alchemy.SymbolicAlchemy],
                           SymbolicAlchemyBot],
    add_trackers_to_env: Callable[[symbolic_alchemy.SymbolicAlchemy], None],
) -> Dict[str, Any]:
  """Runs a symbolic alchemy bot for 1 episode.

  Args:
    episode_items: Named tuple with the fields:
      init_stones - The stones for each trial.
      init_potions - The potions for each trial.
    chemistry: Named tuple with the fields:
      potion_map - The potion map which is actually present in this episode.
      stone_map - The stone map which is actually present in this episode.
      graph - The graph which is actually present in this episode.
    reward_weights: A callable which gives a reward for some stone coords.
    bot_from_env: Callable which returns the bot to run given the environment.
    add_trackers_to_env: Add trackers to the environment.

  Returns:
    The results of running the bot for an episode.
  """

  env = symbolic_alchemy.get_symbolic_alchemy_fixed(
      episode_items, chemistry, reward_weights=reward_weights)
  add_trackers_to_env(env)

  return bot_from_env(env).run_episode()


def get_multi_trial_ideal_observer_reward(
    episode_items: utils.EpisodeItems, chemistry: utils.Chemistry,
    reward_weights: RewardWeights, precomputed: PrecomputedMaps,
    minimise_world_states: bool,
    add_trackers_to_env: Callable[[symbolic_alchemy.SymbolicAlchemy], None],
) -> Dict[str, Any]:
  """Applies a greedy policy using ideal observer reward estimates for n trials.

  Args:
    episode_items: Named tuple with the fields:
      init_stones - The stones for each trial.
      init_potions - The potions for each trial.
    chemistry: Named tuple with the fields:
      potion_map - The potion map which is actually present in this episode.
      stone_map - The stone map which is actually present in this episode.
      graph - The graph which is actually present in this episode.
    reward_weights: A callable which gives a reward for some stone coords.
    precomputed: Precomputed maps used for speed.
    minimise_world_states: Let the objective be to minimise the number of world
      states at the end of the trial instead of to maximise the accumulated
      reward.
    add_trackers_to_env: Add trackers to the environment.

  Returns:
    The reward for each trial.
    An event tracker for each trial.
    A dictionary of extra information to record about how the ideal observer ran
      on each trial.
  """
  def bot_from_env(
      env: symbolic_alchemy.SymbolicAlchemy) -> IdealObserverBot:
    return IdealObserverBot(
        reward_weights, precomputed, env, minimise_world_states)

  return run_symbolic_alchemy_bot(
      episode_items, chemistry, reward_weights, bot_from_env,
      add_trackers_to_env=add_trackers_to_env)
