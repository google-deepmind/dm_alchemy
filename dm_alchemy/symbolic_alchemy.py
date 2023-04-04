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
"""Symbolic environment for alchemy."""

import abc
import copy
import functools
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

from dm_alchemy import event_tracker
from dm_alchemy import symbolic_alchemy_trackers
from dm_alchemy.ideal_observer import precomputed_maps
from dm_alchemy.types import graphs
from dm_alchemy.types import helpers
from dm_alchemy.types import stones_and_potions
from dm_alchemy.types import utils as type_utils
import dm_env
from dm_env import specs
import numpy as np

Stone = stones_and_potions.Stone
Potion = stones_and_potions.Potion
LatentStoneIndex = stones_and_potions.LatentStoneIndex
LatentStone = stones_and_potions.LatentStone
LatentPotion = stones_and_potions.LatentPotion
AlignedStoneIndex = stones_and_potions.AlignedStoneIndex
PerceivedPotionIndex = stones_and_potions.PerceivedPotionIndex
AlignedStone = stones_and_potions.AlignedStone
PerceivedStone = stones_and_potions.PerceivedStone
PerceivedPotion = stones_and_potions.PerceivedPotion
CAULDRON = stones_and_potions.CAULDRON
random_stone_map = stones_and_potions.random_stone_map
random_potion_map = stones_and_potions.random_potion_map
random_latent_stone = stones_and_potions.random_latent_stone
random_latent_potion = stones_and_potions.random_latent_potion
random_rotation = stones_and_potions.random_rotation
random_graph = graphs.random_graph
graph_distr = graphs.graph_distr
possible_constraints = graphs.possible_constraints
bottleneck1_constraints = graphs.bottleneck1_constraints
bottleneck2_constraints = graphs.bottleneck2_constraints
bottleneck3_constraints = graphs.bottleneck3_constraints
no_bottleneck_constraints = graphs.no_bottleneck_constraints
Chemistry = type_utils.Chemistry
TrialItems = type_utils.TrialItems
ElementContent = type_utils.ElementContent
SeeChemistry = type_utils.ChemistrySeen
GetChemistryObs = type_utils.GetChemistryObsFns

SymbolicAlchemyTracker = symbolic_alchemy_trackers.SymbolicAlchemyTracker
ActionInfo = symbolic_alchemy_trackers.ActionInfo

STONE_COUNT_SCALE = 3.0
POTION_COUNT_SCALE = 12.0
POTION_TYPE_SCALE = PerceivedPotion.num_types / 2.0
REWARD_SCALE = stones_and_potions.max_reward()
END_TRIAL = helpers.END_TRIAL
NO_OP = -1
UNKNOWN_TYPE = -3
MAX_STONES = 3
MAX_POTIONS = 12
NO_EDGE = graphs.NO_EDGE
DEFAULT_MAX_STEPS_PER_TRIAL = 20


def int_action_to_tuple(
    action: int, slot_based: bool, end_trial_action: bool
) -> Tuple[int, int]:
  """Converts integer action to tuple.

  In the integer representation, if we have an end trial action the mapping is
  as follows, otherwise subtract 1 from the integers shown below:
    0 represents ending the trial
    1 represents doing nothing
    The remaining integers represent putting a stone into a potion or into the
      cauldron, i.e. s * (num potion types + 1) + 2 represents putting stone
      type s into the cauldron (or stone index s in the slot based version)
      and s * (num potion types + 1) + 3 + p represents putting stone type s
      (or again index s) into potion type p (or index p).
  In the tuple representation:
    (-2, -2) represents ending the trial
    (-1, -1) represents no-op
    (s, -1) represents putting a stone of type (or index) s into the cauldron
    (s, p) represents putting a stone of type (or index) s into a potion of type
      (or index) p

  Args:
    action: Integer representing the action to take.
    slot_based: Whether the action is for a slot based env or type based.
    end_trial_action: Whether we have an end trial action

  Returns:
     Tuple representing the action to take.
  """
  altered_action = copy.deepcopy(action)
  altered_action -= 1
  if end_trial_action:
    altered_action -= 1
  if altered_action < 0:
    return altered_action, altered_action
  if slot_based:
    potions_and_cauldron = MAX_POTIONS + 1
  else:
    potions_and_cauldron = PerceivedPotion.num_types + 1
  return (altered_action // potions_and_cauldron,
          (altered_action % potions_and_cauldron) - 1)


def tuple_action_to_int(
    action: Tuple[int, int], slot_based: bool, end_trial_action: bool
) -> int:
  """Converts tuple action to integer."""
  stone, potion = action
  num_special_actions = 2 if end_trial_action else 1
  if stone < 0:
    return stone + num_special_actions
  if slot_based:
    potions_and_cauldron = MAX_POTIONS + 1
  else:
    potions_and_cauldron = PerceivedPotion.num_types + 1
  return stone * potions_and_cauldron + potion + 1 + num_special_actions


def slot_based_action_to_int(
    action: type_utils.SlotBasedAction, end_trial_action: bool
) -> int:
  """Converts tuple action to integer."""
  num_special_actions = 2 if end_trial_action else 1
  if action.end_trial:
    val = END_TRIAL
  elif action.no_op:
    val = NO_OP
  else:
    potions_and_cauldron = MAX_POTIONS + 1
    if action.cauldron:
      potion_as_int = 0
    else:
      potion_as_int = action.potion_ind + 1
    val = action.stone_ind * potions_and_cauldron + potion_as_int
  return val + num_special_actions


def normalized_poss_dim_map_observation(
    dim_maps: Sequence[Sequence[int]]
) -> List[float]:
  """Gets an observation for which dimension maps are possible."""
  return [0.0 if (list(x) not in dim_maps) else (
      1.0 if len(dim_maps) == 1 else 0.5) for x in
          stones_and_potions.get_all_dim_ordering()]


def normalized_dir_map_observation(
    dir_map: Sequence[int]
) -> List[float]:
  """Normalizes a direction map to be between 0 and 1."""
  return [(x + 1.0) / 2.0 for x in dir_map]


class SymbolicAlchemy(dm_env.Environment, abc.ABC):
  """Symbolic alchemy environment.

  The chemistry and stones and potions are generated using the callables passed
  in. We assume the potion map, stone map, graph, stones and potions can be
  independently generated.

  Currently observations are just the outcome stone index of the action
  performed or -1 if no stone was used. Later observations will be the whole set
  of stones and potions available in some format.
  """

  def __init__(
      self, chemistry_gen: Callable[[], type_utils.Chemistry],
      reward_weights: stones_and_potions.RewardWeights,
      items_gen: Callable[[int], type_utils.TrialItems], num_trials: int,
      end_trial_action: bool = False,
      max_steps_per_trial: int = DEFAULT_MAX_STEPS_PER_TRIAL,
      see_chemistries: Optional[Mapping[str, type_utils.ChemistrySeen]] = None,
      generate_events: bool = False, fix_obs_length: bool = False,
      observe_used: bool = True):
    """Constructs a symbolic alchemy environment.

    Args:
      chemistry_gen: Generate a chemistry for an episode.
      reward_weights: Structure which tells us the reward for a given stone.
      items_gen: Generate a set of stones and potions for a trial in an episode.
      num_trials: The number of trials in each episode.
      end_trial_action: Whether the agent has an action to end the trial early.
      max_steps_per_trial: The number of steps the agent can take before the
        trial is automatically ended.
      see_chemistries: Optional map from name to a structure containing
        information about how to form a chemistry, i.e. which parts and whether
        the content should be ground truth or the belief state. These are added
        to the observation dictionary. If None, then no chemistries are added.
      generate_events: Whether to track items generated and potions and stones
        used and return this information when events is called on the
        environment. This is not necessary during training but is used when we
        run analysis on the environment.
      fix_obs_length: Whether to fix the length of the chemistry observation
        that's fed in as input. If False, will only concatenate parts of the
        chemistry that are supposed to be seen.
      observe_used: Whether to have a feature for each item slot which is set to
        1 if the item is used and 0 otherwise.
    """
    self._is_new_trial = False
    self.observe_used = observe_used
    self._chemistry_gen = chemistry_gen
    self._reward_weights = reward_weights or stones_and_potions.RewardWeights(
        coefficients=[1, 1, 1], offset=0, bonus=12)
    # These are the items generated each trial.
    self._items_gen = items_gen
    self._num_trials = num_trials
    self._chemistry = None
    self.trial_number = -1
    self.game_state: Optional[event_tracker.GameState] = None
    self._is_last_step = True
    self._steps_this_trial = 0
    self.max_steps_per_trial = max_steps_per_trial
    self.trackers: MutableMapping[str, SymbolicAlchemyTracker] = {}
    if generate_events:
      trackers = {
          symbolic_alchemy_trackers.AddMatrixEventTracker.NAME:
              symbolic_alchemy_trackers.AddMatrixEventTracker(),
          symbolic_alchemy_trackers.ItemGeneratedTracker.NAME:
              symbolic_alchemy_trackers.ItemGeneratedTracker()}
      self.add_trackers(trackers)
    self._outcome_tracker = None
    self._end_trial_action = end_trial_action
    self._fix_obs_length = fix_obs_length

    # Whether we can see the ground truth chemistry in observations
    self.see_chemistries = see_chemistries or {}
    precomputeds: List[precomputed_maps.PrecomputedMaps] = []
    for see_chemistry in self.see_chemistries.values():
      see_chemistry.initialise_precomputed()
      if see_chemistry.precomputed is not None:
        precom = see_chemistry.precomputed  # type: precomputed_maps.PrecomputedMaps
        precomputeds.append(precom)
    self._precomputed = precomputeds[0] if precomputeds else None
    self._possible_partial_graph_indices = None
    self._contents = None
    if self._precomputed is not None:
      belief_state_tracker = {
          symbolic_alchemy_trackers.BeliefStateTracker.NAME:
              symbolic_alchemy_trackers.BeliefStateTracker(
                  self._precomputed, self)}
      num_possible_partial_graphs = len(
          self._precomputed.partial_graph_index_to_possible_index)
      self._possible_partial_graph_indices = np.array([
          0 for _ in range(num_possible_partial_graphs)], dtype=int)
      for ind, i in (
          self._precomputed.partial_graph_index_to_possible_index.items()):
        self._possible_partial_graph_indices[i] = ind
      self.add_trackers(belief_state_tracker)

  def add_trackers(
      self, trackers: Mapping[str, SymbolicAlchemyTracker]
  ) -> None:
    self.trackers.update(trackers)

  def events(self) -> Dict[str, Any]:
    """If it is the last step returns events for the episode."""
    events = {}
    if not self._is_last_step:
      return events
    events.update({'chemistry': self._chemistry})
    for tracker_name in ['matrix_event', 'items_generated']:
      if tracker_name in self.trackers:
        events.update({
            tracker_name: self.trackers[tracker_name].episode_returns()})
    return events

  def _new_trial(self) -> None:
    self._steps_this_trial = 0
    if self.trial_number + 1 >= self._num_trials:
      self.trial_number = -1
      self._is_last_step = True
    else:
      self._is_new_trial = True
      self.trial_number += 1
      items = self._items_gen(self.trial_number)
      reward_tracker = event_tracker.RewardTracker(self._reward_weights)
      self.game_state = event_tracker.GameState(
          self._chemistry.graph, trial_items=items,
          event_trackers=[reward_tracker, self._outcome_tracker])
      self.trial_start(self.game_state)

  def reset_no_observation(self) -> None:
    self.trial_number = -1
    self._is_last_step = False
    # Generate a chemistry for this episode.
    self._chemistry = self._chemistry_gen()
    # At the start of the episode sample what the contents of each element of
    # the chemistry observation will be.
    self._contents = {k: see_chemistry.sample_contents()
                      for k, see_chemistry in self.see_chemistries.items()}
    self._outcome_tracker = event_tracker.LatestOutcomeTracker(
        self._chemistry.potion_map, self._chemistry.stone_map,
        self._chemistry.rotation)
    self.episode_start(self._chemistry)
    self._new_trial()

  def reset(self) -> dm_env.TimeStep:
    self.reset_no_observation()
    return dm_env.TimeStep(
        dm_env.StepType.FIRST, None, None, self.observation())

  def step_no_observation(
      self, action: type_utils.SlotBasedAction,
      original_action: Optional[Union[
          type_utils.SlotBasedAction, type_utils.TypeBasedAction]] = None
  ) -> Optional[float]:
    """Takes a step in the environment without producing an observation.

    Args:
      action: The action to take in integer representation.
      original_action: The original action in whatever form which may not be
        runnable passed in for tracking.

    Returns:
      The reward gained this step or None if we must reset.
    """
    if self._is_last_step:
      self.reset_no_observation()
      return None
    self._is_new_trial = False
    reward = 0
    self.game_state.trackers['latest_outcome'].reset()

    self._steps_this_trial += 1
    if action.using_stone:
      reward_start = self.game_state.trackers['reward'].reward
      if action.cauldron:
        self.game_state.stone_used(action.stone_ind)
      elif action.using_potion:
        self.game_state.potion_used(action.stone_ind, action.potion_ind)
      else:
        # Need to call this for outcome tracker to know what the stone type
        # is.
        self.game_state.failed_potion_use(action.stone_ind)
      reward = self.game_state.trackers['reward'].reward - reward_start

    type_based_action = (
        self.game_state.trackers['latest_outcome'].type_based_action or
        type_utils.TypeBasedAction(
            end_trial=action.end_trial, no_op=action.no_op))
    self.action_and_outcome(
        action=type_based_action,
        outcome=self.game_state.trackers['latest_outcome'].outcome,
        action_info=ActionInfo(
            original_action, action.using_stone, action.using_potion))

    if action.end_trial or self._steps_this_trial >= self.max_steps_per_trial:
      # If the current stone is -1 then end the trial.
      self.trial_end()
      self._new_trial()

    return float(reward)

  def step(self, action: int) -> dm_env.TimeStep:
    """Takes a step in the environment using the action passed in.

    Args:
      action: The action to take in integer representation.

    Returns:
      A timestep with the observation, reward, step type and discount.
    """
    return self.step_slot_based_action(self._int_to_slot_based_action(action))

  def step_slot_based_action(
      self, action: type_utils.SlotBasedAction
  ) -> dm_env.TimeStep:
    return self._internal_step(self._runnable_slot_based_action(action), action)

  def step_type_based_action(
      self, action: type_utils.TypeBasedAction
  ) -> dm_env.TimeStep:
    """Takes a step in the environment using a slot based action."""
    return self._internal_step(self._type_based_to_slot_based(action), action)

  def _internal_step(
      self, action: type_utils.SlotBasedAction,
      original_action: Union[
          type_utils.SlotBasedAction, type_utils.TypeBasedAction]
  ) -> dm_env.TimeStep:
    """Takes a step in the environment using a slot based action."""
    if not self._end_trial_action and action.end_trial:
      raise ValueError('Env has no end trial action')
    return self.construct_step(self.step_no_observation(
        action, original_action))

  def _type_based_to_slot_based(
      self, action: type_utils.TypeBasedAction
  ) -> type_utils.SlotBasedAction:
    stone_ind, potion_ind = None, None
    if action.using_stone:
      aligned_stone = stones_and_potions.align(
          action.perceived_stone, self._chemistry.rotation)
      latent_stone = self._chemistry.stone_map.apply(aligned_stone)
      stone_ind = self.game_state.get_stone_ind(stone=graphs.Node(
          -1, latent_stone.latent_coords))
    if action.using_potion:
      latent_potion = self._chemistry.potion_map.apply(action.perceived_potion)
      potion_ind = self.game_state.get_potion_ind(potion=latent_potion)
    return type_utils.SlotBasedAction(
        end_trial=action.end_trial, no_op=action.no_op, stone_ind=stone_ind,
        cauldron=action.cauldron, potion_ind=potion_ind)

  def construct_step(
      self, reward: Optional[float], discount: Optional[float] = 1.0
  ) -> dm_env.TimeStep:
    if reward is None:
      # If reward is None this is the first step of an episode.
      step_type = dm_env.StepType.FIRST
      discount = None
    elif self._is_last_step:
      step_type = dm_env.StepType.LAST
      # There should be no rewards considered beyond the last step.
      discount = 0.0
    else:
      step_type = dm_env.StepType.MID

    return dm_env.TimeStep(step_type, reward, discount, self.observation())

  def _int_to_slot_based_action(
      self, action: int
  ) -> type_utils.SlotBasedAction:
    """Converts integer action to simplified action.

    In the integer representation, if we have an end trial action the mapping is
    as follows, otherwise subtract 1 from the integers shown below:
      0 represents ending the trial
      1 represents doing nothing
      The remaining integers represent putting a stone into a potion or into the
        cauldron, i.e. s * (num potion types + 1) + 2 represents putting stone
        type s into the cauldron (or stone index s in the slot based version)
        and s * (num potion types + 1) + 3 + p represents putting stone type s
        (or again index s) into potion type p (or index p).

    Args:
      action: Integer representing the action to take.

    Returns:
       SlotBasedAction representing the action to take.
    """
    altered_action = copy.deepcopy(action)
    altered_action -= 1
    if self._end_trial_action:
      altered_action -= 1
    if altered_action < 0:
      return type_utils.SlotBasedAction(
          end_trial=altered_action == END_TRIAL, no_op=altered_action == NO_OP)
    potions_and_cauldron = MAX_POTIONS + 1
    stone_ind = altered_action // potions_and_cauldron
    potion_ind = (altered_action % potions_and_cauldron) - 1
    if potion_ind < 0:
      return type_utils.SlotBasedAction(
          stone_ind=stone_ind, cauldron=True)
    return type_utils.SlotBasedAction(
        stone_ind=stone_ind,
        potion_ind=potion_ind)

  def _slot_based_action_to_int(
      self, action: type_utils.SlotBasedAction
  ) -> int:
    """Converts tuple action to integer."""
    return slot_based_action_to_int(action, self._end_trial_action)

  def _runnable_slot_based_action(
      self, action: type_utils.SlotBasedAction
  ) -> type_utils.SlotBasedAction:
    new_action = copy.deepcopy(action)
    if action.stone_ind is not None and not self.game_state.has_stone_ind(
        action.stone_ind):
      new_action.stone_ind = None
    if action.potion_ind is not None and not self.game_state.has_potion_ind(
        action.potion_ind):
      new_action.potion_ind = None
    return new_action

  def observation_spec(self):
    num_stone_features, num_potion_features = slot_based_num_features(
        self.observe_used)
    obs_features = ((num_stone_features * MAX_STONES) +
                    (num_potion_features * MAX_POTIONS))
    obs_spec = {
        'symbolic_obs': specs.Array(shape=(obs_features,), dtype=np.float32,
                                    name='symbolic_obs')}
    obs_spec.update(self.chem_observation_spec())
    return obs_spec

  def action_spec(self):
    # Actions for each stone slot in each potion slot, each stone slot in the
    # cauldron, end trial and no-op.
    num_special_actions = 2 if self._end_trial_action else 1
    num_actions = MAX_STONES * (MAX_POTIONS + 1) + num_special_actions
    return (specs.BoundedArray(
        shape=(), dtype=int, minimum=0, maximum=num_actions - 1,
        name='action'))

  def _num_features(self):
    return slot_based_num_features(self.observe_used)

  def _default_features(self):
    num_axes = stones_and_potions.get_num_axes()
    stone_features = [2 for _ in range(num_axes + 1)]
    potion_features = [1]
    if self.observe_used:
      # Set used to 1 by default, we will set it to 0 for items that exist.
      stone_features.append(1)
      potion_features.append(1)
    return (np.array([stone_features], dtype=np.float32),
            np.array([potion_features], dtype=np.float32))

  def observation(self):
    # If we are using the slot based representation then get features for each
    # stone which is present.
    num_axes = stones_and_potions.get_num_axes()
    default_stone_features, default_potion_features = self._default_features()
    stone_features = np.concatenate(
        [default_stone_features for _ in range(MAX_STONES)], axis=0)
    potion_features = np.concatenate(
        [default_potion_features for _ in range(MAX_POTIONS)], axis=0)
    existing_stones = (self.game_state.existing_stones() if self.game_state
                       else [])
    existing_potions = (self.game_state.existing_potions() if self.game_state
                        else [])
    for stone in existing_stones:
      stone_ind = self.game_state.get_stone_ind(stone_inst=stone.idx)
      assert 0 <= stone_ind < MAX_STONES, 'stone idx out of range'
      aligned_stone = self._chemistry.stone_map.apply_inverse(
          stone.latent_stone())
      perceived_stone = stones_and_potions.unalign(
          aligned_stone, self._chemistry.rotation)
      for f in range(num_axes):
        stone_features[stone_ind, f] = perceived_stone.perceived_coords[f]
      # This feature is equivalent to the value indicator seen on the stone as
      # it distinguishes different reward values.
      stone_features[stone_ind, num_axes] = (
          perceived_stone.reward / stones_and_potions.max_reward())
      if self.observe_used:
        stone_features[stone_ind, num_axes + 1] = 0.0
    for potion in existing_potions:
      potion_ind = self.game_state.get_potion_ind(potion_inst=potion.idx)
      assert potion_ind < MAX_POTIONS, 'potion idx out of range'
      latent_potion = potion.latent_potion()
      perceived_potion = self._chemistry.potion_map.apply_inverse(latent_potion)
      potion_features[potion_ind, 0] = (
          (perceived_potion.index() / POTION_TYPE_SCALE) - 1.0)
      if self.observe_used:
        potion_features[potion_ind, 1] = 0.0

    concat_obs = {'symbolic_obs': np.concatenate(
        (stone_features.reshape((-1,)), potion_features.reshape((-1,))))}
    concat_obs.update(self.chem_observation())
    return concat_obs

  def _rotation_known(self) -> bool:
    belief_state_tracker: symbolic_alchemy_trackers.BeliefStateTracker = (
        self.trackers['belief_state'])
    return len(belief_state_tracker.belief_state.possible_rotations) == 1

  def get_belief_state_edge_vals(
      self, unknown_edge_vals: List[float]
  ) -> List[float]:
    if not self._rotation_known():
      return unknown_edge_vals
    # First attempt - do the simplest thing of setting all unknown to 0.5
    # and otherwise set to 0 or 1
    belief_state_tracker: symbolic_alchemy_trackers.BeliefStateTracker = (
        self.trackers['belief_state'])
    this_adjmat = belief_state_tracker.get_partial_graph(
        self._possible_partial_graph_indices).known_adj_mat.astype(
            np.float32)
    this_adjmat[this_adjmat == helpers.UNKNOWN] = 0.5
    return graphs.edge_values_from_adj_mat(this_adjmat)

  def get_ground_truth_edge_vals(self) -> List[float]:
    # Get adjacency matrix corresponding to current graph
    this_adjmat = graphs.convert_graph_to_adj_mat(self._chemistry.graph).astype(
        np.float32)
    this_adjmat[this_adjmat != NO_EDGE] = 1.0
    return graphs.edge_values_from_adj_mat(this_adjmat)

  def get_belief_state_potion_map_obs(
      self, unknown_potion_map: List[float]
  ) -> List[float]:
    if not self._rotation_known():
      return unknown_potion_map
    belief_state_tracker = self.trackers['belief_state']  # type: symbolic_alchemy_trackers.BeliefStateTracker
    partial_potion_map = belief_state_tracker.get_partial_potion_map(
        self._precomputed.index_to_perm_index)
    potion_map_possible = (
        belief_state_tracker.belief_state.belief_state.world_state_distribution.
        potion_map_possible)
    dim_maps = [stones_and_potions.potion_map_from_index(
        p, self._precomputed.index_to_perm_index).dim_map
                for p in potion_map_possible]
    dir_map = [0 if d == helpers.UNKNOWN else d
               for d in partial_potion_map.dir_map]
    return (normalized_poss_dim_map_observation(dim_maps) +
            normalized_dir_map_observation(dir_map))

  def get_ground_truth_potion_map_obs(self) -> List[float]:
    dim_maps = [self._chemistry.potion_map.dim_map]
    dir_map = self._chemistry.potion_map.dir_map
    return (normalized_poss_dim_map_observation(dim_maps) +
            normalized_dir_map_observation(dir_map))

  def get_belief_state_stone_map_obs(
      self, unknown_stone_map: List[float]
  ) -> List[float]:
    if not self._rotation_known():
      return unknown_stone_map
    belief_state_tracker: symbolic_alchemy_trackers.BeliefStateTracker = (
        self.trackers['belief_state'])
    partial_stone_map = belief_state_tracker.get_partial_stone_map()
    return normalized_dir_map_observation([
        0 if d == helpers.UNKNOWN else d
        for d in partial_stone_map.latent_pos_dir])

  def get_ground_truth_stone_map_obs(self) -> List[float]:
    return normalized_dir_map_observation(
        self._chemistry.stone_map.latent_pos_dir)

  def get_belief_state_rotation(
      self, unknown_rotation: List[float]
  ) -> List[float]:
    if not self._rotation_known():
      return unknown_rotation
    return self.get_ground_truth_rotation()

  def get_ground_truth_rotation(self) -> List[float]:
    return [1.0 if stones_and_potions.rotations_equal(
        self._chemistry.rotation, rotation) else 0.0
            for rotation in stones_and_potions.possible_rotations()]

  def chem_observation(self) -> Dict[str, np.ndarray]:
    """Converts the ground truth chemistry/mappings into observation vector."""
    # full representation of chemistry should be length 28
    unknown_potion_obs = [0.5 for _ in range(
        len(stones_and_potions.get_all_dim_ordering()) +
        stones_and_potions.get_num_axes())]
    unknown_stone_obs = [0.5 for _ in range(stones_and_potions.get_num_axes())]
    unknown_edge_vals = [0.5 for _ in range(graphs.num_edges_in_cube())]
    unknown_rotation = [0.5 for _ in range(len(
        stones_and_potions.possible_rotations()))]
    # If we try to get an observation before we have contents (i.e. before the
    # environment is reset) we return unknown values. This happens when we are
    # running a 3d environment and we cannot reset the symbolic environment
    # until the 3d environment sends messages containing the chemistry and
    # items.
    contents = (self._contents or
                {k: [ElementContent.UNKNOWN] * len(see_chemistry.groups)
                 for k, see_chemistry in self.see_chemistries.items()})
    get_obs = type_utils.GetChemistryObsFns(
        potion_map={
            ElementContent.UNKNOWN: lambda: unknown_potion_obs,
            ElementContent.GROUND_TRUTH: functools.partial(
                SymbolicAlchemy.get_ground_truth_potion_map_obs, self),
            ElementContent.BELIEF_STATE: functools.partial(
                SymbolicAlchemy.get_belief_state_potion_map_obs, self,
                unknown_potion_obs)},
        stone_map={ElementContent.UNKNOWN: lambda: unknown_stone_obs,
                   ElementContent.GROUND_TRUTH: functools.partial(
                       SymbolicAlchemy.get_ground_truth_stone_map_obs, self),
                   ElementContent.BELIEF_STATE: functools.partial(
                       SymbolicAlchemy.get_belief_state_stone_map_obs, self,
                       unknown_stone_obs)},
        graph={ElementContent.UNKNOWN: lambda: unknown_edge_vals,
               ElementContent.GROUND_TRUTH: functools.partial(
                   SymbolicAlchemy.get_ground_truth_edge_vals, self),
               ElementContent.BELIEF_STATE: functools.partial(
                   SymbolicAlchemy.get_belief_state_edge_vals, self,
                   unknown_edge_vals)},
        rotation={ElementContent.UNKNOWN: lambda: unknown_rotation,
                  ElementContent.GROUND_TRUTH: functools.partial(
                      SymbolicAlchemy.get_ground_truth_rotation, self),
                  ElementContent.BELIEF_STATE: functools.partial(
                      SymbolicAlchemy.get_belief_state_rotation, self,
                      unknown_rotation)})
    return {k: np.array(see_chemistry.form_observation(contents[k], get_obs),
                        dtype=np.float32)
            for k, see_chemistry in self.see_chemistries.items()}

  def chem_observation_spec(self) -> Dict[str, specs.Array]:
    return {k: specs.Array(shape=(see_chemistry.obs_size(),), dtype=np.float32,
                           name=k)
            for k, see_chemistry in self.see_chemistries.items()}

  def step_spec(self) -> None:
    raise NotImplementedError

  def is_new_trial(self) -> bool:
    return self._is_new_trial

  def is_last_step(self) -> bool:
    return self._is_last_step

  def perceived_stone(self, stone: Stone) -> PerceivedStone:
    aligned_stone = self._chemistry.stone_map.apply_inverse(
        stone.latent_stone())
    perceived_stone = stones_and_potions.unalign(
        aligned_stone, self._chemistry.rotation)
    return perceived_stone

  def perceived_stones(self) -> List[PerceivedStone]:
    return [self.perceived_stone(s) for s in self.game_state.existing_stones()]

  def perceived_potion(self, potion: Potion) -> PerceivedPotion:
    return self._chemistry.potion_map.apply_inverse(
        potion.latent_potion())

  def perceived_potions(self) -> List[PerceivedPotion]:
    return [self.perceived_potion(s)
            for s in self.game_state.existing_potions()]

  def use_positive_stones(self) -> dm_env.TimeStep:
    overall_reward = 0
    overall_discount = 1.0
    # If it is the last step of an episode reset to start a new one.
    if self._is_last_step:
      self.reset()
      # Reward and discount will be None as we have started a new episode.
    pos_stone_inds = self.game_state.get_stones_above_thresh(
        self._reward_weights, threshold=0)
    for stone_ind in pos_stone_inds:
      timestep = self.step_slot_based_action(type_utils.SlotBasedAction(
          stone_ind=stone_ind, cauldron=True))
      overall_reward += timestep.reward
      overall_discount *= timestep.discount
      if self._is_last_step or self.is_new_trial():
        return self.construct_step(float(overall_reward), overall_discount)
    end_trial_reward, end_trial_discount = self.end_trial()
    overall_reward += end_trial_reward
    overall_discount *= end_trial_discount

    # Get the cumulative reward and discount and the final step type and
    # observation.
    return self.construct_step(float(overall_reward), overall_discount)

  def end_trial(self) -> Tuple[float, float]:
    overall_reward = 0.0
    overall_discount = 1.0
    # If it is the last step of an episode reset to start a new one.
    if self._is_last_step:
      self.reset()
      # Reward and discount will be None as we have started a new episode.
    if self._end_trial_action:
      reward = self.step_no_observation(type_utils.SlotBasedAction(
          end_trial=True))
      overall_reward += reward
    else:
      # If it is a new trial take at least one step.
      if self.is_new_trial():
        reward = self.step_no_observation(type_utils.SlotBasedAction(
            no_op=True))
        overall_reward += reward
      while not (self._is_last_step or self.is_new_trial()):
        reward = self.step_no_observation(type_utils.SlotBasedAction(
            no_op=True))
        overall_reward += reward
    return overall_reward, overall_discount

  def episode_start(self, chemistry: type_utils.Chemistry) -> None:
    for tracker in self.trackers.values():
      tracker.episode_start(chemistry)

  def trial_start(self, game_state: event_tracker.GameState) -> None:
    for tracker in self.trackers.values():
      tracker.trial_start(game_state)

  def action_and_outcome(
      self, action: type_utils.TypeBasedAction,
      outcome: Optional[PerceivedStone], action_info: ActionInfo
  ) -> None:
    for tracker in self.trackers.values():
      tracker.action_and_outcome(action, outcome, action_info)

  def trial_end(self) -> None:
    for tracker in self.trackers.values():
      tracker.trial_end()

  def episode_returns(self) -> Dict[str, Any]:
    returns = {}
    for name, tracker in self.trackers.items():
      returns.update({name: tracker.episode_returns()})
    return returns


def slot_based_num_features(observe_used: bool) -> Tuple[int, int]:
  num_axes = stones_and_potions.get_num_axes()
  num_stone_features = num_axes + 1
  num_potion_features = 1
  if observe_used:
    num_stone_features += 1
    num_potion_features += 1
  return num_stone_features, num_potion_features


def slot_based_stone_feature_dims(observe_used, rotated_stone_positions):
  """Gets the dimensions of the observation containing each stone feature."""
  stone_features, _ = slot_based_num_features(observe_used)
  potion_section = MAX_STONES * stone_features
  poss_vals = [-1, 0, 1] if rotated_stone_positions else [-1, 1]
  return ((slice(i, potion_section, stone_features)
           for i in range(stones_and_potions.get_num_axes())),
          np.array(poss_vals, dtype=float))


def slot_based_stone_reward_dims(observe_used):
  """Gets the dimensions of the observation containing stone rewards."""
  stone_features, _ = slot_based_num_features(observe_used)
  potion_section = MAX_STONES * stone_features
  return (slice(
      stones_and_potions.get_num_axes(), potion_section, stone_features),
          np.array([r / REWARD_SCALE for r in stones_and_potions.POSS_REWARDS],
                   dtype=float))


def slot_based_potion_colour_dims(observe_used):
  """Gets the dimensions of the observation containing potion colours."""
  stone_features, potion_features = slot_based_num_features(observe_used)
  potion_section = MAX_STONES * stone_features
  return (slice(potion_section, None, potion_features),
          np.array([(p / POTION_TYPE_SCALE) - 1.0
                    for p in range(PerceivedPotion.num_types)], dtype=float))


def take_simplified_action(
    simplified_action: Union[
        type_utils.SlotBasedAction, type_utils.TypeBasedAction],
    env: SymbolicAlchemy
) -> dm_env.TimeStep:
  """Takes action from the simplified action spec."""
  # In the simplified action spec the agent can end the trial and use all
  # positive stones with a single action.
  if simplified_action.end_trial:
    return env.use_positive_stones()
  if isinstance(simplified_action, type_utils.SlotBasedAction):
    return env.step_slot_based_action(simplified_action)
  return env.step_type_based_action(simplified_action)


def get_symbolic_alchemy_level(
    level_name, observe_used=True, end_trial_action=False,
    num_trials=10, num_stones_per_trial=3, num_potions_per_trial=12, seed=None,
    reward_weights=None, max_steps_per_trial=DEFAULT_MAX_STEPS_PER_TRIAL,
    see_chemistries=None, generate_events=False):
  """Gets a symbolic alchemy instance of the level passed in."""
  random_state = np.random.RandomState(seed)

  if 'perceptual_mapping_randomized' in level_name:
    _, index_to_perm_index = precomputed_maps.get_perm_index_conversion()
    stone_map_gen = functools.partial(
        random_stone_map, random_state=random_state)
    seeded_rand_potion_map = functools.partial(
        random_potion_map, random_state=random_state)
    potion_map_gen = lambda: seeded_rand_potion_map(index_to_perm_index)
  else:
    stone_map_gen = stones_and_potions.all_fixed_stone_map
    potion_map_gen = stones_and_potions.all_fixed_potion_map

  seeded_rand_graph = functools.partial(
      random_graph, random_state=random_state)
  if 'random_bottleneck' in level_name:
    graph_gen = lambda: seeded_rand_graph(graph_distr(possible_constraints()))
  elif 'bottleneck1' in level_name:
    graph_gen = (
        lambda: seeded_rand_graph(graph_distr(bottleneck1_constraints())))
  elif 'bottleneck2' in level_name:
    graph_gen = (
        lambda: seeded_rand_graph(graph_distr(bottleneck2_constraints())))
  elif 'bottleneck3' in level_name:
    graph_gen = (
        lambda: seeded_rand_graph(graph_distr(bottleneck3_constraints())))
  else:
    graph_gen = (
        lambda: seeded_rand_graph(graph_distr(no_bottleneck_constraints())))

  if 'rotation' in level_name:
    rotation_gen = functools.partial(random_rotation, random_state=random_state)
  else:
    rotation_gen = lambda: np.eye(3)

  def items_gen(unused_trial_number):
    del unused_trial_number
    stones_in_trial = [random_latent_stone(random_state=random_state)
                       for _ in range(num_stones_per_trial)]
    potions_in_trial = [random_latent_potion(random_state=random_state)
                        for _ in range(num_potions_per_trial)]
    return TrialItems(potions=potions_in_trial, stones=stones_in_trial)

  def chemistry_gen():
    return Chemistry(
        potion_map_gen(), stone_map_gen(), graph_gen(), rotation_gen())

  return SymbolicAlchemy(
      observe_used=observe_used,
      chemistry_gen=chemistry_gen,
      reward_weights=reward_weights,
      items_gen=items_gen,
      num_trials=num_trials,
      end_trial_action=end_trial_action,
      max_steps_per_trial=max_steps_per_trial,
      see_chemistries=see_chemistries,
      generate_events=generate_events)


def get_symbolic_alchemy_fixed(
    episode_items, chemistry, observe_used=True, reward_weights=None,
    end_trial_action=False, max_steps_per_trial=DEFAULT_MAX_STEPS_PER_TRIAL,
    see_chemistries=None, generate_events=False):
  """Symbolic alchemy which generates same chemistry and items every episode."""

  return SymbolicAlchemy(
      observe_used=observe_used,
      chemistry_gen=lambda: chemistry,
      reward_weights=reward_weights,
      items_gen=lambda i: episode_items.trials[i],
      num_trials=episode_items.num_trials,
      end_trial_action=end_trial_action,
      max_steps_per_trial=max_steps_per_trial,
      see_chemistries=see_chemistries,
      generate_events=generate_events)
