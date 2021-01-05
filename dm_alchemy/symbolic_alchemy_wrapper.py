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
"""Wrapper for a 3d alchemy to keep a symbolic alchemy in sync."""

from dm_alchemy import get_meta_data
from dm_alchemy import symbolic_alchemy
from dm_alchemy.types import event_unpacking
from dm_alchemy.types import stones_and_potions
from dm_alchemy.types import unity_python_conversion
from dm_alchemy.types import utils
import dm_env


def _add_to_obs(obs, to_add, name):
  if isinstance(obs, tuple):
    return obs + (to_add,)
  if isinstance(obs, dict):
    obs[name] = to_add
    return obs
  if isinstance(obs, list):
    return obs + [to_add]
  # If it is not already a tuple, dict or list, then make it a tuple.
  return obs, to_add


class SymbolicAlchemyWrapper(dm_env.Environment):
  """Take a 3d alchemy environment and keep a symbolic env in sync with it."""

  def __init__(
      self, env3d, level_name, see_chemistries=None,
      see_symbolic_observation=False):
    self.env3d = env3d
    value_coefficients, value_offset, _, bonus, _ = get_meta_data.to_meta_data(
        level_name)
    reward_weights = stones_and_potions.RewardWeights(
        coefficients=value_coefficients, offset=value_offset, bonus=bonus)

    self.env_symbolic = symbolic_alchemy.SymbolicAlchemy(
        chemistry_gen=lambda: self.chemistry,
        reward_weights=reward_weights,
        items_gen=lambda unused_trial_number: self.items,
        num_trials=10,
        see_chemistries=see_chemistries,
        observe_used=True,
    )
    self.items = utils.TrialItems(stones=[], potions=[])
    self._perceived_stones = []
    self._perceived_potions = []
    self.chemistry = None
    self.see_symbolic_observation = see_symbolic_observation
    self._trial_in_progress = False
    self._trial_has_started = False

  def process_step_events(self, events):
    for event in events:
      if 'TrialEnded' in event.name:
        self._trial_has_started = False
        self.items = utils.TrialItems(stones=[], potions=[])
        self._perceived_stones = []
        self._perceived_potions = []
      elif 'TrialStarted' in event.name:
        self._trial_has_started = True
        # At this point we should have all stones and potions and the chemistry.
        aligned_stones = [
            stones_and_potions.align(stone, self.chemistry.rotation)
            for stone, _ in self._perceived_stones]
        latent_stones = [self.chemistry.stone_map.apply(stone)
                         for stone in aligned_stones]
        stones = [
            stones_and_potions.Stone(i, stone.latent_coords)
            for (_, i), stone in zip(self._perceived_stones, latent_stones)]
        latent_potions = [self.chemistry.potion_map.apply(potion)
                          for potion, _ in self._perceived_potions]
        potions = [
            stones_and_potions.Potion(i, potion.latent_dim, potion.latent_dir)
            for (_, i), potion in zip(self._perceived_potions, latent_potions)]
        self.items = utils.TrialItems(stones=stones, potions=potions)

        # When we get an event saying that the new trial has started in the 3d
        # version it should be safe to end the previous trial in the symbolic
        # version.
        if self._trial_in_progress:
          self.env_symbolic.end_trial()
        if self.env_symbolic.is_last_step():
          self.env_symbolic.reset()
        # Once the first trial is started there is always a trial in progress
        # from then on.
        self._trial_in_progress = True
      elif 'PotionUsed' in event.name:
        potion_inst_id, stone_inst_id = event_unpacking.unpack_potion_used(
            event)
        stone_ind = self.env_symbolic.game_state.get_stone_ind(
            stone_inst=stone_inst_id)
        potion_ind = self.env_symbolic.game_state.get_potion_ind(
            potion_inst=potion_inst_id)
        # Take an action putting the stone in the potion.
        self.env_symbolic.step_slot_based_action(utils.SlotBasedAction(
            stone_ind=stone_ind, potion_ind=potion_ind))
      elif 'StoneUsed' in event.name:
        stone_inst_id = event_unpacking.unpack_stone_used(event)
        stone_ind = self.env_symbolic.game_state.get_stone_ind(
            stone_inst=stone_inst_id)
        # Take an action putting the stone in the cauldron.
        self.env_symbolic.step_slot_based_action(utils.SlotBasedAction(
            stone_ind=stone_ind, cauldron=True))
      elif 'ChemistryCreated' in event.name:
        chem, rot = event_unpacking.unpack_chemistry_and_rotation(event)
        self.chemistry = unity_python_conversion.from_unity_chemistry(chem, rot)
      else:
        potions = event_unpacking.get_potions([event])
        stones = event_unpacking.get_stones([event])
        if (potions or stones) and self._trial_has_started:
          self.items = utils.TrialItems(stones=[], potions=[])
          self._perceived_stones = []
          self._perceived_potions = []
          self._trial_has_started = False
        self._perceived_potions.extend(potions)
        self._perceived_stones.extend(stones)

  def step(self, action) -> dm_env.TimeStep:
    timestep = self.env3d.step(action)
    # If a symbolic action has occurred take the action in the symbolic
    # environment.
    self.process_step_events(self.env3d.events())
    return self.add_observations(timestep)

  def reset(self) -> dm_env.TimeStep:
    timestep = self.env3d.reset()
    self.items = utils.TrialItems(stones=[], potions=[])
    self._perceived_stones = []
    self._perceived_potions = []
    self._trial_has_started = False
    self.process_step_events(self.env3d.events())
    return self.add_observations(timestep)

  def add_observations(self, timestep):
    new_observation = timestep.observation
    symbolic_observation = self.env_symbolic.observation()
    if self.see_symbolic_observation:
      new_observation = _add_to_obs(
          new_observation, symbolic_observation['symbolic_obs'], 'symbolic_obs')
    for name in self.env_symbolic.see_chemistries.keys():
      new_observation = _add_to_obs(
          new_observation, symbolic_observation[name], name)
    return dm_env.TimeStep(
        step_type=timestep.step_type, reward=timestep.reward,
        discount=timestep.discount, observation=new_observation)

  def observation_spec(self):
    obs_spec = self.env3d.observation_spec()
    if self.see_symbolic_observation:
      symbolic_obs = self.env_symbolic.observation_spec()['symbolic_obs']
      obs_spec = _add_to_obs(obs_spec, symbolic_obs, 'symbolic_obs')
    for name in self.env_symbolic.see_chemistries.keys():
      chem_obs_spec = self.env_symbolic.observation_spec()[name]
      obs_spec = _add_to_obs(obs_spec, chem_obs_spec, name)
    return obs_spec

  def action_spec(self):
    return self.env3d.action_spec()

  # Forward other attribute lookups to the 3d environment.
  def __getattr__(self, name):
    return getattr(self.env3d, name)
