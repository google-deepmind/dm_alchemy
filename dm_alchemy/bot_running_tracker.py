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
"""Symbolic alchemy tracker which runs bots in sync with environment."""
import copy
from typing import Any, Callable, List, Optional, Tuple

from dm_alchemy import symbolic_alchemy
from dm_alchemy import symbolic_alchemy_bots
from dm_alchemy import symbolic_alchemy_trackers
from dm_alchemy.types import stones_and_potions
from dm_alchemy.types import utils
import numpy as np
import tree


AlignedStoneIndex = stones_and_potions.AlignedStoneIndex
PerceivedPotionIndex = stones_and_potions.PerceivedPotionIndex
PerceivedStone = stones_and_potions.PerceivedStone


def get_envs_and_bots(
    env: symbolic_alchemy.SymbolicAlchemy,
    bot_from_env: Callable[[symbolic_alchemy.SymbolicAlchemy],
                           symbolic_alchemy_bots.SymbolicAlchemyBot],
    num_bots: int,
    add_trackers_to_env: Callable[[symbolic_alchemy.SymbolicAlchemy], None]
) -> Tuple[List[symbolic_alchemy.SymbolicAlchemy],
           List[symbolic_alchemy_bots.SymbolicAlchemyBot]]:
  """Gets several copies of the environment and bots to run on them.

  Args:
    env: The base environment to make copies of.
    bot_from_env: A callable which given an environment creates a bot to run on
      it.
    num_bots: The number of bots and environment copies to make.
    add_trackers_to_env: A callable which adds any trackers required to the
      environment copies.

  Returns:
    A list of copies of the environment.
    A list of bots to run on copies of the environment.
  """
  env_without_bot_running_trackers = copy.deepcopy(env)
  if env_without_bot_running_trackers:
    env_without_bot_running_trackers.trackers = {}
    add_trackers_to_env(env_without_bot_running_trackers)
  envs = [copy.deepcopy(env_without_bot_running_trackers)
          for _ in range(num_bots)]
  bots = [bot_from_env(e) for e in envs]
  return envs, bots


class BotRunningTracker(symbolic_alchemy_trackers.SymbolicAlchemyTracker):
  """Run bots which take actions whenever an action is taken in the environment."""

  NAME = 'bot_runner'

  @property
  def name(self) -> str:
    return self.NAME

  def __init__(
      self, env: symbolic_alchemy.SymbolicAlchemy,
      bot_from_env: Callable[[symbolic_alchemy.SymbolicAlchemy],
                             symbolic_alchemy_bots.SymbolicAlchemyBot],
      num_bots: int,
      add_trackers_to_env: Callable[[symbolic_alchemy.SymbolicAlchemy], None]):
    self.envs, self.bots = get_envs_and_bots(
        env, bot_from_env, num_bots, add_trackers_to_env)

  def episode_start(self, unused_chemistry: utils.Chemistry) -> None:
    """Resets all environments when an episode has started."""
    for env in self.envs:
      env.reset()

  def action_and_outcome(
      self, action: utils.TypeBasedAction,
      unused_outcome: Optional[PerceivedStone],
      unused_action_info: symbolic_alchemy_trackers.ActionInfo
  ) -> None:
    """Lets bots take an action when an action is taken in the main environment."""
    del unused_outcome, unused_action_info
    # Only take an action if a potion is used (actions to put stones in the
    # cauldron will happen automatically when the end trial action is selected).
    if action.using_potion:
      for bot, env in zip(self.bots, self.envs):
        new_action = bot.select_action()
        # Don't select end trial actions as this will be done when the trial
        # ends in the original environment.
        if new_action.end_trial:
          continue
        symbolic_alchemy.take_simplified_action(new_action, env)

  def trial_end(self) -> None:
    """Ends the trial in all copies of the environment."""
    for env in self.envs:
      symbolic_alchemy.take_simplified_action(
          utils.SlotBasedAction(end_trial=True), env)

  def episode_returns(self) -> Any:
    """Gets returns from trackers on environment copies."""
    return tree.map_structure(lambda *args: np.mean(args, axis=0), *tuple(
        env.episode_returns() for env in self.envs))

  def default_returns(self, num_trials: int, num_actions_per_trial: int) -> Any:
    """Gets a set of default returns."""
    ep_returns = self.envs[0].episode_returns()
    def to_float(arg: Any) -> Any:
      if isinstance(arg, np.ndarray):
        return arg.astype(np.float)
      return float(arg)

    return tree.map_structure(to_float, ep_returns)
