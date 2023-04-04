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
"""Tests for the ideal observer."""

import random

from absl.testing import absltest
from absl.testing import parameterized
from dm_alchemy import symbolic_alchemy
from dm_alchemy.types import graphs
from dm_alchemy.types import stones_and_potions
from dm_alchemy.types import utils
from dm_env import test_utils
import numpy as np

Stone = stones_and_potions.Stone
Potion = stones_and_potions.Potion
LatentStone = stones_and_potions.LatentStone
LatentPotion = stones_and_potions.LatentPotion
AlignedStone = stones_and_potions.AlignedStone
PerceivedPotion = stones_and_potions.PerceivedPotion
PerceivedStone = stones_and_potions.PerceivedStone
CAULDRON = stones_and_potions.CAULDRON


_NUM_TRIALS = 2
_TRIAL_STONES = [Stone(0, [-1, -1, 1]),
                 Stone(1, [1, 1, 1]),
                 Stone(2, [1, 1, -1])]
_TEST_STONES = [_TRIAL_STONES for _ in range(_NUM_TRIALS)]

_TRIAL_POTIONS = [Potion(0, 1, 1),
                  Potion(1, 1, -1),
                  Potion(2, 1, 1),
                  Potion(3, 1, 1),
                  Potion(4, 2, 1),
                  Potion(5, 1, 1),
                  Potion(6, 2, -1),
                  Potion(7, 0, 1),
                  Potion(8, 2, -1),
                  Potion(9, 2, 1),
                  Potion(10, 1, 1),
                  Potion(11, 1, -1)]
_TEST_POTIONS = [_TRIAL_POTIONS for _ in range(_NUM_TRIALS)]

_MAX_STEPS_PER_TRIAL = 20

_FIXED_POTION_MAP = stones_and_potions.all_fixed_potion_map()
_FIXED_STONE_MAP = stones_and_potions.all_fixed_stone_map()
_FIXED_ROTATION = np.eye(3, dtype=int)

_CHEM_NAME = 'test_chem'


def random_slot_based_action():
  stone_ind = random.randint(0, symbolic_alchemy.MAX_STONES - 1)
  potion_ind = random.randint(-1, symbolic_alchemy.MAX_POTIONS - 1)
  if potion_ind < 0:
    return utils.SlotBasedAction(stone_ind=stone_ind, cauldron=True)
  return utils.SlotBasedAction(stone_ind=stone_ind, potion_ind=potion_ind)


def reward_fcn():
  return stones_and_potions.RewardWeights([1, 1, 1], 0, 12)


def make_fixed_chem_env(
    constraint=None, potion_map=_FIXED_POTION_MAP, stone_map=_FIXED_STONE_MAP,
    rotation=_FIXED_ROTATION, test_stones=None, test_potions=None, **kwargs):
  if constraint is None:
    constraint = graphs.no_bottleneck_constraints()[0]
  env = symbolic_alchemy.get_symbolic_alchemy_fixed(
      episode_items=utils.EpisodeItems(
          potions=test_potions or _TEST_POTIONS,
          stones=test_stones or _TEST_STONES),
      chemistry=utils.Chemistry(
          graph=graphs.create_graph_from_constraint(constraint),
          potion_map=potion_map, stone_map=stone_map, rotation=rotation),
      reward_weights=reward_fcn(), max_steps_per_trial=_MAX_STEPS_PER_TRIAL,
      **kwargs)
  return env


def make_random_chem_env(**kwargs):
  env = symbolic_alchemy.get_symbolic_alchemy_level(
      level_name='perceptual_mapping_randomized_with_random_bottleneck',
      reward_weights=reward_fcn(), max_steps_per_trial=_MAX_STEPS_PER_TRIAL,
      **kwargs)
  return env


def make_random_action_sequence(num_trials, end_trial_action):
  num_random_actions = 10
  assert num_random_actions <= _MAX_STEPS_PER_TRIAL
  # On each trial take some random actions then end the trial.
  actions = []
  for _ in range(num_trials):
    # Create random actions, some of which may not be possible.
    actions.extend(
        [random_slot_based_action() for _ in range(num_random_actions)])
    # End the trial
    if end_trial_action:
      actions.append(utils.SlotBasedAction(end_trial=True))
    else:
      for _ in range(_MAX_STEPS_PER_TRIAL - num_random_actions):
        actions.append(utils.SlotBasedAction(no_op=True))
  return [symbolic_alchemy.slot_based_action_to_int(action, end_trial_action)
          for action in actions]


def type_based_use_stone(env, perceived_stone, unused_stone):
  del unused_stone
  return env.step_type_based_action(utils.TypeBasedAction(
      stone=perceived_stone, cauldron=True))


def slot_based_use_stone(env, unused_perceived_stone, stone):
  del unused_perceived_stone
  return env.step_slot_based_action(utils.SlotBasedAction(
      stone_ind=stone.idx, cauldron=True))


def type_based_use_potion(
    env, perceived_stone, unused_stone, perceived_potion, unused_potion):
  del unused_stone, unused_potion
  return env.step_type_based_action(utils.TypeBasedAction(
      stone=perceived_stone, potion=perceived_potion))


def slot_based_use_potion(
    env, unused_perceived_stone, stone, unused_perceived_potion, potion):
  del unused_perceived_stone, unused_perceived_potion
  return env.step_slot_based_action(utils.SlotBasedAction(
      stone_ind=stone.idx, potion_ind=potion.idx))


class SymbolicAlchemyTest(test_utils.EnvironmentTestMixin):

  num_trials = _NUM_TRIALS

  def test_no_op(self):
    env = self.make_object_under_test()
    timestep = env.reset()
    # Perform a no-op
    new_timestep = env.step_slot_based_action(utils.SlotBasedAction(no_op=True))
    np.testing.assert_allclose(timestep.observation['symbolic_obs'],
                               new_timestep.observation['symbolic_obs'])
    self.assertEqual(new_timestep.reward, 0)

  def env_mid_trial(self, reset=True, no_op_steps=0):
    env = self.make_object_under_test()
    if reset:
      env.reset()
    for _ in range(no_op_steps):
      env.step_slot_based_action(utils.SlotBasedAction(no_op=True))
    return env

  def end_trial_test(self, reset=True, no_op_steps=0):
    env = self.env_mid_trial(reset=reset, no_op_steps=no_op_steps)
    self.assertEqual(env.trial_number, 0 if reset else -1)
    env.end_trial()
    self.assertEqual(env.trial_number, 1)

  def test_end_trial(self):
    # parameterised tests in base class do not work
    # end trial straight away
    self.end_trial_test(reset=False)
    # end trial after each number of no ops from 0 to max steps per trial - 1.
    # Note if we take all of max steps the trial will end before we call
    # end_trial.
    for no_op_steps in range(_MAX_STEPS_PER_TRIAL):
      self.end_trial_test(no_op_steps=no_op_steps)


class SymbolicAlchemyFixedChemTest(SymbolicAlchemyTest):
  """Test symbolic alchemy using the mixin."""

  def use_pos_stones_test(self, expected_reward, reset=True, no_op_steps=0):
    env = self.env_mid_trial(reset=reset, no_op_steps=no_op_steps)
    timestep = env.use_positive_stones()
    self.assertAlmostEqual(timestep.reward, expected_reward, 4)

  def test_use_positive_stones(self):
    # parameterised tests in base class do not work
    # end trial straight away
    self.use_pos_stones_test(expected_reward=16.0, reset=False)
    # use positive stones after each number of no ops from 0 to
    # max steps per trial - 1. For the last one we will only get a reward of 15
    # as there is only time to use 1 stone.
    for no_op_steps in range(_MAX_STEPS_PER_TRIAL):
      exp_reward = 15.0 if no_op_steps == _MAX_STEPS_PER_TRIAL - 1 else 16.0
      self.use_pos_stones_test(
          expected_reward=exp_reward, no_op_steps=no_op_steps)

  def _test_use_stone(self, take_action):
    env = self.make_object_under_test()
    env.reset()
    num_stone_features, _ = symbolic_alchemy.slot_based_num_features(
        env.observe_used)
    default_stone_features, _ = env._default_features()
    latent_stones = [stone.latent_stone() for stone in _TEST_STONES[0]]
    aligned_stones = [_FIXED_STONE_MAP.apply_inverse(stone)
                      for stone in latent_stones]
    perceived_stones = [stones_and_potions.unalign(stone, _FIXED_ROTATION)
                        for stone in aligned_stones]
    for stone, perceived_stone, latent_stone in zip(
        _TEST_STONES[0], perceived_stones, latent_stones):
      new_timestep = take_action(env, perceived_stone, stone)
      expected_reward = reward_fcn()(latent_stone.latent_coords)
      self.assertEqual(new_timestep.reward, expected_reward)
      # Observation should be set to the default.
      stone_obs = new_timestep.observation['symbolic_obs'][
          num_stone_features * stone.idx:num_stone_features * (stone.idx + 1)]
      for stone_feat, default_stone_feat in zip(
          stone_obs, default_stone_features[0, :]):
        self.assertAlmostEqual(stone_feat, default_stone_feat, 4)

    # After using the stones end the trial
    end_trial_reward, _ = env.end_trial()
    self.assertEqual(end_trial_reward, 0)

  def test_use_stone(self):
    self._test_use_stone(slot_based_use_stone)
    self._test_use_stone(type_based_use_stone)

  def _test_use_potion(self, take_action):
    env = self.make_object_under_test()
    env.reset()
    stone = _TEST_STONES[0][0]
    potion = _TEST_POTIONS[0][0]
    aligned_stone = _FIXED_STONE_MAP.apply_inverse(stone.latent_stone())
    perceived_stone = stones_and_potions.unalign(aligned_stone, _FIXED_ROTATION)
    perceived_potion = _FIXED_POTION_MAP.apply_inverse(potion.latent_potion())
    new_timestep = take_action(
        env, perceived_stone, stone, perceived_potion, potion)
    self.assertEqual(new_timestep.reward, 0)

    stone_features, _ = symbolic_alchemy.slot_based_num_features(
        env.observe_used)
    potion_start_index = stone_features * symbolic_alchemy.MAX_STONES
    potion0_obs = new_timestep.observation['symbolic_obs'][potion_start_index]
    self.assertAlmostEqual(potion0_obs, 1.0, 4)
    stone_obs = new_timestep.observation['symbolic_obs'][:stone_features]
    # Coords change to -1, 1, 1 and reward changes to 1/max reward
    self.assertAlmostEqual(stone_obs[0], -1.0, 4)
    self.assertAlmostEqual(stone_obs[1], 1.0, 4)
    self.assertAlmostEqual(stone_obs[2], 1.0, 4)
    self.assertAlmostEqual(
        stone_obs[3], 1.0 / stones_and_potions.max_reward(), 4)

    # After using the potion end the trial
    end_trial_reward, _ = env.end_trial()
    self.assertEqual(end_trial_reward, 0)

  def test_use_potion(self):
    self._test_use_potion(slot_based_use_potion)
    self._test_use_potion(type_based_use_potion)

  def make_object_under_test(self):
    """Make an environment which will be tested by the mixin."""
    return make_fixed_chem_env(
        observe_used=self.observe_used, end_trial_action=self.end_trial_action)

  def make_action_sequence(self):
    return make_random_action_sequence(self.num_trials, self.end_trial_action)

  def test_initial_observation(self):
    # Type based observation should have scaled count for each type
    env = self.make_object_under_test()
    timestep = env.reset()
    # All fixed so the perceptual mapping is the identity
    num_axes = stones_and_potions.get_num_axes()
    stone_features, potion_features = symbolic_alchemy.slot_based_num_features(
        env.observe_used)
    for stone in _TEST_STONES[0]:
      # The features should be the perceptual features then the scaled reward
      stone_obs = timestep.observation['symbolic_obs'][
          stone_features * stone.idx:stone_features * (stone.idx + 1)]
      for dim in range(num_axes):
        self.assertAlmostEqual(stone_obs[dim], stone.latent[dim], 4)
      self.assertAlmostEqual(
          stone_obs[num_axes],
          sum(stone.latent)/stones_and_potions.max_reward(), 4)
      if self.observe_used:
        self.assertAlmostEqual(stone_obs[num_axes + 1], 0.0, 4)
    # Test that 2 potions of the same type are observed the same, and potions of
    # different types are observed different.
    # 0 and 2 are the same, 1 is different.
    potion_start_index = stone_features * symbolic_alchemy.MAX_STONES
    potion_obs = []
    for i in range(symbolic_alchemy.MAX_POTIONS):
      feat_start = potion_start_index + i * potion_features
      feat_end = feat_start + potion_features
      potion_obs.append(
          timestep.observation['symbolic_obs'][feat_start:feat_end])
    self.assertAlmostEqual(potion_obs[0][0], potion_obs[2][0], 4)
    self.assertNotAlmostEqual(potion_obs[0][0], potion_obs[1][0], 4)
    if self.observe_used:
      self.assertAlmostEqual(potion_obs[0][1], 0.0, 4)


class SymbolicAlchemyFixedChemObserveUsedTest(
    SymbolicAlchemyFixedChemTest):

  observe_used = True


class SymbolicAlchemyFixedChemObserveUsedEndTrialTest(
    SymbolicAlchemyFixedChemObserveUsedTest, absltest.TestCase):

  end_trial_action = True


class SymbolicAlchemyFixedChemObserveUsedNoEndTrialTest(
    SymbolicAlchemyFixedChemObserveUsedTest, absltest.TestCase):

  end_trial_action = False


class SymbolicAlchemyFixedChemNoObserveUsedTest(
    SymbolicAlchemyFixedChemTest):

  observe_used = False


class SymbolicAlchemyFixedChemNoObserveUsedEndTrialTest(
    SymbolicAlchemyFixedChemNoObserveUsedTest, absltest.TestCase):

  end_trial_action = True


class SymbolicAlchemyFixedChemNoObserveUsedNoEndTrialTest(
    SymbolicAlchemyFixedChemNoObserveUsedTest, absltest.TestCase):

  end_trial_action = False


class SymbolicAlchemyRandomChemTest(SymbolicAlchemyTest):
  """Test symbolic alchemy with random chem each episode using the mixin."""

  def make_object_under_test(self, **kwargs):
    """Make an environment which will be tested by the mixin."""
    return make_random_chem_env(
        observe_used=self.observe_used, end_trial_action=self.end_trial_action,
        num_trials=self.num_trials, **kwargs)

  def make_action_sequence(self):
    return make_random_action_sequence(
        self.num_trials, self.end_trial_action)

  def test_seed(self):
    env1 = self.make_object_under_test(seed=0)
    env1.reset()
    env2 = self.make_object_under_test(seed=0)
    env2.reset()
    self.assertEqual(graphs.constraint_from_graph(env1._chemistry.graph),
                     graphs.constraint_from_graph(env2._chemistry.graph))
    self.assertEqual(env1._chemistry.potion_map, env2._chemistry.potion_map)
    self.assertEqual(env1._chemistry.stone_map, env2._chemistry.stone_map)
    self.assertEqual(env1.game_state.existing_items(),
                     env2.game_state.existing_items())


class SymbolicAlchemyRandomChemObserveUsedTest(
    SymbolicAlchemyRandomChemTest):

  observe_used = True


class SymbolicAlchemyRandomChemObserveUsedEndTrialTest(
    SymbolicAlchemyRandomChemObserveUsedTest, absltest.TestCase):

  end_trial_action = True


class SymbolicAlchemyRandomChemObserveUsedNoEndTrialTest(
    SymbolicAlchemyRandomChemObserveUsedTest, absltest.TestCase):

  end_trial_action = False


class SymbolicAlchemyRandomChemNoObserveUsedTest(
    SymbolicAlchemyRandomChemTest):

  observe_used = False


class SymbolicAlchemyRandomChemNoObserveUsedEndTrialTest(
    SymbolicAlchemyRandomChemNoObserveUsedTest, absltest.TestCase):

  end_trial_action = True


class SymbolicAlchemyRandomChemNoObserveUsedNoEndTrialTest(
    SymbolicAlchemyRandomChemNoObserveUsedTest, absltest.TestCase):

  end_trial_action = False


class SymbolicAlchemySeeChemistryTest(parameterized.TestCase):
  """We don't do the full mixin tests for the chemistry observation."""

  def _make_env(self, see_chemistry, constraint, **kwargs):
    return make_fixed_chem_env(
        constraint=constraint,
        see_chemistries={_CHEM_NAME: see_chemistry},
        observe_used=True, end_trial_action=False, **kwargs)

  @parameterized.parameters(
      # In the graph observations edges are in the following order:
      #     _________11__________
      #    /|                  /|
      #  9/ |               10/ |
      #  /  |                /  |
      # /___|_____8_________/   |
      # |   |6              |   |7
      # |   |               |   |
      # |2  |               |4  |
      # |   |_______5_______|___|
      # |   /               |   /
      # |  /1               |  /3
      # | /                 | /
      # |/________0_________|/
      #
      # With coordinate system:
      # |
      # |z  /
      # |  /y
      # | /
      # |/___x___
      #
      {'see_chemistry': utils.ChemistrySeen(
          potion_map=utils.PotionMapElement(present=False),
          stone_map=utils.StoneMapElement(present=False),
          rotation=utils.RotationElement(present=False),
          content=utils.ElementContent.GROUND_TRUTH),
       'constraint': graphs.no_bottleneck_constraints()[0],
       # With no constraints all edges should be present
       'expected_obs': np.ones((12,), np.float32),
       'expected_len': 12},
      {'see_chemistry': utils.ChemistrySeen(
          potion_map=utils.PotionMapElement(present=False),
          stone_map=utils.StoneMapElement(present=False),
          rotation=utils.RotationElement(present=False),
          content=utils.ElementContent.GROUND_TRUTH),
       'constraint': graphs.bottleneck1_constraints()[0],
       # For bottleneck1 constraint the only x direction edge that exists is 8,
       # so 0, 5 and 11 are missing.
       'expected_obs': np.array([0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0,
                                 1.0, 1.0, 0.0], np.float32),
       'expected_len': 12},
      {'see_chemistry': utils.ChemistrySeen(
          stone_map=utils.StoneMapElement(present=False),
          graph=utils.GraphElement(present=False),
          rotation=utils.RotationElement(present=False),
          content=utils.ElementContent.GROUND_TRUTH),
       'constraint': graphs.no_bottleneck_constraints()[0],
       # First 6 entries are a 1-hot for the dimension map, in this case the
       # dimension map used is the first one.
       # The next 3 entries are 0 or 1 for the direction map, in this case all
       # directions are positive.
       'expected_obs': np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                                np.float32),
       'expected_len': 9},
      {'see_chemistry': utils.ChemistrySeen(
          potion_map=utils.PotionMapElement(present=False),
          graph=utils.GraphElement(present=False),
          rotation=utils.RotationElement(present=False),
          content=utils.ElementContent.GROUND_TRUTH),
       'constraint': graphs.no_bottleneck_constraints()[0],
       # 3 entries are 0 or 1 for the direction map, in this case all directions
       # are positive.
       'expected_obs': np.array([1.0, 1.0, 1.0], np.float32),
       'expected_len': 3},
      {'see_chemistry': utils.ChemistrySeen(
          content=utils.ElementContent.GROUND_TRUTH),
       'constraint': graphs.no_bottleneck_constraints()[0],
       # Observations are from the previous tests concatenated with graph first,
       # then potion map then stone map.
       'expected_obs': np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # graph
                                 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # graph
                                 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # potion dim map
                                 1.0, 1.0, 1.0,  # potion dir map
                                 1.0, 1.0, 1.0,  # stone map
                                 1.0, 0.0, 0.0, 0.0], np.float32),  # rotation
       'expected_len': 28},
      # Tests for the belief state observation.
      {'see_chemistry': utils.ChemistrySeen(
          potion_map=utils.PotionMapElement(present=False),
          stone_map=utils.StoneMapElement(present=False),
          rotation=utils.RotationElement(present=False),
          content=utils.ElementContent.BELIEF_STATE,
          precomputed='perceptual_mapping_randomized_with_random_bottleneck'),
       'constraint': graphs.no_bottleneck_constraints()[0],
       # With no actions the belief state should be unknown for all edges.
       'expected_obs': 0.5 * np.ones((12,), np.float32),
       'expected_len': 12},
      {'see_chemistry': utils.ChemistrySeen(
          potion_map=utils.PotionMapElement(present=False),
          stone_map=utils.StoneMapElement(present=False),
          rotation=utils.RotationElement(present=False),
          content=utils.ElementContent.BELIEF_STATE,
          precomputed='perceptual_mapping_randomized_with_random_bottleneck'),
       'constraint': graphs.bottleneck1_constraints()[0],
       # It shouldn't make a difference whether the underlying chemistry has a
       # constraint or not everythin is unknown.
       'expected_obs': 0.5 * np.ones((12,), np.float32),
       'expected_len': 12},
      {'see_chemistry': utils.ChemistrySeen(
          stone_map=utils.StoneMapElement(present=False),
          graph=utils.GraphElement(present=False),
          rotation=utils.RotationElement(present=False),
          content=utils.ElementContent.BELIEF_STATE,
          precomputed='perceptual_mapping_randomized_with_random_bottleneck'),
       'constraint': graphs.no_bottleneck_constraints()[0],
       # First 6 entries are a 1-hot for the dimension map, with no actions all
       # of the dimesnsion maps are possible so the entries are all unknown.
       # The next 3 entries are 0 or 1 for the direction map, or 0.5 for unknown
       # which is the case if no actions are taken.
       'expected_obs': 0.5 * np.ones((9,), np.float32),
       'expected_len': 9},
      {'see_chemistry': utils.ChemistrySeen(
          potion_map=utils.PotionMapElement(present=False),
          graph=utils.GraphElement(present=False),
          rotation=utils.RotationElement(present=False),
          content=utils.ElementContent.BELIEF_STATE,
          precomputed='perceptual_mapping_randomized_with_random_bottleneck'),
       'constraint': graphs.no_bottleneck_constraints()[0],
       # 3 entries are 0 or 1 for the direction map, in this case all directions
       # are positive, since the test stones include an instance of the best
       # stone, the stone map should be known from the start.
       'expected_obs': np.array([1.0, 1.0, 1.0], np.float32),
       'expected_len': 3},
      {'see_chemistry': utils.ChemistrySeen(
          content=utils.ElementContent.BELIEF_STATE,
          rotation=utils.RotationElement(present=False),
          precomputed='perceptual_mapping_randomized_with_random_bottleneck'),
       'constraint': graphs.no_bottleneck_constraints()[0],
       # Observations are from the previous tests concatenated with graph first,
       # then potion map then stone map.
       'expected_obs': np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5,  # graph
                                 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,  # graph
                                 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,  # potion dim map
                                 0.5, 0.5, 0.5,  # potion dir map
                                 1.0, 1.0, 1.0], np.float32),  # stone map
       'expected_len': 24},
      {'see_chemistry': utils.ChemistrySeen(
          content=utils.ElementContent.BELIEF_STATE,
          rotation=utils.RotationElement(present=False),
          precomputed='perceptual_mapping_randomized_with_random_bottleneck'),
       'constraint': graphs.no_bottleneck_constraints()[0],
       'actions': [utils.SlotBasedAction(stone_ind=0, potion_ind=0)],
       # If we put the 0th stone into the 0th potion we will see a change on
       # axis 1, we will become certain that the dim map is either [0, 1, 2] or
       # [2, 1, 0], we will become certain that the edge from (-1, -1, 1) to
       # (-1, 1, 1) exists.
       'expected_obs': np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5,  # graph
                                 0.5, 0.5, 0.5, 1.0, 0.5, 0.5,  # graph
                                 0.5, 0.0, 0.0, 0.0, 0.0, 0.5,  # potion dim map
                                 0.5, 1.0, 0.5,  # potion dir map
                                 1.0, 1.0, 1.0], np.float32),  # stone map
       'expected_len': 24},
      # Tests for a combination of content types
      {'see_chemistry': utils.ChemistrySeen(
          potion_map=utils.PotionMapElement(present=False),
          stone_map=utils.StoneMapElement(present=False),
          rotation=utils.RotationElement(present=False),
          groups=[
              utils.GroupInChemistry(
                  {utils.ElementType.GRAPH: {0, 1, 2, 3}}, [1.0, 0.0, 0.0]),
              utils.GroupInChemistry(
                  {utils.ElementType.GRAPH: {4, 5, 6}}, [0.0, 0.0, 1.0]),
              utils.GroupInChemistry(
                  {utils.ElementType.GRAPH: {7, 8, 9, 10, 11}}, [0.0, 1.0, 0.0]),
              ],
          precomputed='perceptual_mapping_randomized_with_random_bottleneck'),
       'constraint': graphs.bottleneck1_constraints()[0],
       'actions': [utils.SlotBasedAction(stone_ind=0, potion_ind=0)],
       # With no actions the belief state should be unknown for all edges.
       'expected_obs': np.array(
           [0.0, 1.0, 1.0, 1.0,  # ground truth - 0 missing 1, 2, 3 exist
            0.5, 0.5, 0.5,  # unknown - these are set to 0.5
            # belief state - after the action 9 is known, others are unknown
            0.5, 0.5, 1.0, 0.5, 0.5], np.float32),
       'expected_len': 12},
      # Rotation tests
      {'see_chemistry': utils.ChemistrySeen(
          potion_map=utils.PotionMapElement(present=False),
          stone_map=utils.StoneMapElement(present=False),
          graph=utils.GraphElement(present=False),
          content=utils.ElementContent.GROUND_TRUTH),
       'constraint': graphs.no_bottleneck_constraints()[0],
       'make_env_kwargs': {
           'rotation': stones_and_potions.rotation_from_angles([0, 0, 0])},
       'expected_obs': np.array([1.0, 0.0, 0.0, 0.0], np.float32),
       'expected_len': 4},
      {'see_chemistry': utils.ChemistrySeen(
          potion_map=utils.PotionMapElement(present=False),
          stone_map=utils.StoneMapElement(present=False),
          graph=utils.GraphElement(present=False),
          content=utils.ElementContent.GROUND_TRUTH),
       'constraint': graphs.no_bottleneck_constraints()[0],
       'make_env_kwargs': {
           'rotation': stones_and_potions.rotation_from_angles([0, 0, -45])},
       'expected_obs': np.array([0.0, 1.0, 0.0, 0.0], np.float32),
       'expected_len': 4},
      {'see_chemistry': utils.ChemistrySeen(
          potion_map=utils.PotionMapElement(present=False),
          stone_map=utils.StoneMapElement(present=False),
          graph=utils.GraphElement(present=False),
          content=utils.ElementContent.GROUND_TRUTH),
       'constraint': graphs.no_bottleneck_constraints()[0],
       'make_env_kwargs': {
           'rotation': stones_and_potions.rotation_from_angles([0, -45, 0])},
       'expected_obs': np.array([0.0, 0.0, 1.0, 0.0], np.float32),
       'expected_len': 4},
      {'see_chemistry': utils.ChemistrySeen(
          potion_map=utils.PotionMapElement(present=False),
          stone_map=utils.StoneMapElement(present=False),
          graph=utils.GraphElement(present=False),
          content=utils.ElementContent.GROUND_TRUTH),
       'constraint': graphs.no_bottleneck_constraints()[0],
       'make_env_kwargs': {
           'rotation': stones_and_potions.rotation_from_angles([-45, 0, 0])},
       'expected_obs': np.array([0.0, 0.0, 0.0, 1.0], np.float32),
       'expected_len': 4},
      # In belief state if we have stones which are unique to a particular
      # rotation then the rotation should be known and possibly part of the
      # stone map.
      {'see_chemistry': utils.ChemistrySeen(
          potion_map=utils.PotionMapElement(present=False),
          graph=utils.GraphElement(present=False),
          content=utils.ElementContent.BELIEF_STATE,
          precomputed=('perceptual_mapping_randomized_with_rotation_and_'
                       'random_bottleneck')),
       'constraint': graphs.no_bottleneck_constraints()[0],
       'make_env_kwargs': {
           'rotation': stones_and_potions.rotation_from_angles([-45, 0, 0]),
           'test_stones': [[Stone(0, [1, 1, 1]), Stone(0, [1, 1, -1])]]},
       'expected_obs': np.array(
           [1.0, 1.0, 1.0,  # stone map
            0.0, 0.0, 0.0, 1.0], np.float32),  # rotation
       'expected_len': 7},
      # Otherwise rotation and stone map observations should both be unknown.
      {'see_chemistry': utils.ChemistrySeen(
          potion_map=utils.PotionMapElement(present=False),
          graph=utils.GraphElement(present=False),
          content=utils.ElementContent.BELIEF_STATE,
          precomputed=('perceptual_mapping_randomized_with_rotation_and_'
                       'random_bottleneck')),
       'constraint': graphs.no_bottleneck_constraints()[0],
       'make_env_kwargs': {
           'rotation': stones_and_potions.rotation_from_angles([-45, 0, 0]),
           'test_stones': [[Stone(0, [1, 1, 1])]]},
       'expected_obs': np.array(
           [0.5, 0.5, 0.5,  # stone map
            0.5, 0.5, 0.5, 0.5], np.float32),  # rotation
       'expected_len': 7},
      {'see_chemistry': utils.ChemistrySeen(
          potion_map=utils.PotionMapElement(present=False),
          graph=utils.GraphElement(present=False),
          content=utils.ElementContent.BELIEF_STATE,
          precomputed=('perceptual_mapping_randomized_with_rotation_and_'
                       'random_bottleneck')),
       'constraint': graphs.no_bottleneck_constraints()[0],
       'make_env_kwargs': {
           'rotation': stones_and_potions.rotation_from_angles([-45, 0, 0]),
           'test_stones': [[Stone(0, [1, 1, 1])]]},
       'actions': [utils.SlotBasedAction(stone_ind=0, potion_ind=6)],
       'expected_obs': np.array(
           [1.0, 1.0, 1.0,  # stone map
            0.0, 0.0, 0.0, 1.0], np.float32),  # rotation
       'expected_len': 7},
  )
  def test_see_chemistry(
      self, see_chemistry, constraint, expected_obs, expected_len,
      actions=None, make_env_kwargs=None):
    """Test the ground truth chemistry observations."""
    env = self._make_env(
        see_chemistry=see_chemistry, constraint=constraint,
        **(make_env_kwargs or {}))
    timestep = env.reset()
    if actions:
      for action in actions:
        timestep = env.step_slot_based_action(action)

    np.testing.assert_allclose(
        timestep.observation[_CHEM_NAME], expected_obs)
    self.assertLen(
        timestep.observation[_CHEM_NAME], expected_len)

  def test_see_chem_before_reset(self):
    env = self._make_env(
        see_chemistry=utils.ChemistrySeen(
            content=utils.ElementContent.GROUND_TRUTH),
        constraint=graphs.no_bottleneck_constraints()[0])
    obs = env.observation()
    # Observation should be all unknown because we have not reset the
    # environment yet.
    np.testing.assert_allclose(obs[_CHEM_NAME], [0.5] * 28)
    # After resetting none of the chem should be unknown.
    env.reset()
    obs = env.observation()
    np.testing.assert_array_less(
        0.01 * np.ones((28,)),
        np.abs(obs[_CHEM_NAME] - np.array([0.5] * 28)))


if __name__ == '__main__':
  absltest.main()
