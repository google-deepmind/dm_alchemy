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

import collections
import functools

from absl.testing import absltest
from absl.testing import parameterized
from dm_alchemy import event_tracker as ev_trk
from dm_alchemy import symbolic_alchemy_bots
from dm_alchemy import symbolic_alchemy_trackers
from dm_alchemy.ideal_observer import ideal_observer
from dm_alchemy.ideal_observer import precomputed_maps
from dm_alchemy.types import graphs
from dm_alchemy.types import stones_and_potions
from dm_alchemy.types import utils
import numpy as np

Counter = collections.Counter
AlignedStone = stones_and_potions.AlignedStone
PerceivedPotion = stones_and_potions.PerceivedPotion
PotionMap = stones_and_potions.PotionMap
StoneMap = stones_and_potions.StoneMap
Stone = stones_and_potions.Stone
Potion = stones_and_potions.Potion

AddMatrixEventTracker = symbolic_alchemy_trackers.AddMatrixEventTracker
ScoreTracker = symbolic_alchemy_trackers.ScoreTracker
BeliefStateTracker = symbolic_alchemy_trackers.BeliefStateTracker


_ALL_FIXED_POTION_MAP = stones_and_potions.all_fixed_potion_map()
_ALL_FIXED_STONE_MAP = stones_and_potions.all_fixed_stone_map()
_ALL_FIXED_GRAPH = graphs.create_graph_from_constraint(
    graphs.no_bottleneck_constraints()[0])
_BOTTLENECK1_CONSTRAINT = graphs.bottleneck1_constraints()[0]
_BOTTLENECK1_GRAPH = graphs.create_graph_from_constraint(
    _BOTTLENECK1_CONSTRAINT)


def add_trackers_to_env(env, reward_weights, precomputed, init_belief_state):
  env.add_trackers({
      AddMatrixEventTracker.NAME: AddMatrixEventTracker(),
      ScoreTracker.NAME: ScoreTracker(reward_weights),
      BeliefStateTracker.NAME: BeliefStateTracker(
          precomputed, env, init_belief_state)})


class IdealObserverTest(parameterized.TestCase):

  level_name_to_precomputed = None

  @classmethod
  def setUpClass(cls):
    super(IdealObserverTest, cls).setUpClass()
    # Load all of the precomputed maps required for all tests otherwise loading
    # them in the tests can take too long and cause the test to time out.
    levels_used = ['all_fixed_bottleneck1', 'perceptual_mapping_randomized',
                   'perceptual_mapping_randomized_with_random_bottleneck']
    cls.level_name_to_precomputed = {
        s: precomputed_maps.load_from_level_name(s) for s in levels_used}

  @parameterized.parameters(
      # 1 stone worth 3, no potions - reward should be 3
      (Counter({AlignedStone(3, np.array([1, 1, 1])): 1}),
       Counter(),
       3),
      # 1 stone worth 1, no potions - reward should be 1
      (Counter({AlignedStone(1, np.array([1, 1, 1])): 1}),
       Counter(),
       1),
      # 1 stone worth -1, no potions - reward should be 0 (since we can choose
      # not to put the stone in)
      (Counter({AlignedStone(-1, np.array([1, 1, 1])): 1}),
       Counter(),
       0),
      # 1 stone worth 3, any number of potions - reward should be 3
      (Counter({AlignedStone(3, np.array([1, 1, 1])): 1}),
       Counter({PerceivedPotion(0, 1): 1}),
       3),
      (Counter({AlignedStone(3, np.array([1, 1, 1])): 1}),
       Counter({PerceivedPotion(0, 1): 1, PerceivedPotion(0, -1): 1}),
       3),
      # If graph has no bottlenecks and we start with a stone worth 1 and
      # potions which go in opposite directions by using a potion we get
      # expected reward of 4/3 since:
      #  1/6 prob of going to 3 then cashing in - contributes 1/2
      #  1/3 prob of going to -1 then we can definitely get back to 1 using the
      #    other potion - contributes 1/3
      #  1/2 prob of staying at 1 then it is best not to use the other potion -
      #    contributes 1/2
      (Counter({AlignedStone(1, np.array([1, 1, 1])): 1}),
       Counter({PerceivedPotion(0, 1): 1, PerceivedPotion(0, -1): 1}),
       1.33333333,
       0,
       'perceptual_mapping_randomized'),
      # If graph has no bottlenecks and we start with a stone worth -1 and
      # potions on different axes then by using a potion we get
      # expected reward of 0.583333 since:
      #  1/6 prob of going to -3 - contributes 0
      #  1/3 prob of going to 1 then if we use the other potion we know it
      #    applies to a different axis so:
      #      1/4 prob of going to 3 - contributes (1/3) * (1/4) * 3 = 1/4
      #      1/4 prob of going to -1 - contributes 0
      #      1/2 prob of staying at 1 - contributes (1/3) * (1/2) * 1 = 1/6
      #  1/2 prob of staying at -1 then the other potion could apply to any axis
      #    so it gives:
      #      1/6 prob of going to -3 - contributes 0
      #      1/3 prob of going to 1 - contributes (1/2) * (1/3) * 1 = 1/6
      #      1/2 prob of staying at -1 - contributes 0
      (Counter({AlignedStone(-1, np.array([1, 1, 1])): 1}),
       Counter({PerceivedPotion(0, 1): 1, PerceivedPotion(1, 1): 1}),
       0.583333,
       0,
       'perceptual_mapping_randomized')
  )
  def test_expected_reward(
      self, aligned_stones, perceived_potions, expected_expected_reward,
      bonus=0, level_name='perceptual_mapping_randomized_with_random_bottleneck'
  ):

    precomputed = IdealObserverTest.level_name_to_precomputed[level_name]

    # Make an initial game state using first trial information.
    current_game_state = ideal_observer.BeliefState(precomputed)

    # Start with no search results.
    search_results = {}

    aligned_stones_ind = collections.Counter({
        k.index(): v for k, v in aligned_stones.items()})
    perceived_potions_ind = collections.Counter({
        k.index(): v for k, v in perceived_potions.items()})
    current_game_state.new_trial(aligned_stones_ind, perceived_potions_ind)

    _, objective, _ = ideal_observer.ideal_observer(
        current_game_state, search_results, bonus, precomputed, False)
    expected_reward, _ = objective
    self.assertAlmostEqual(expected_reward, expected_expected_reward, 4)

  @parameterized.parameters(
      # In the first trial we have 1 stone worth -3 and the 3 potions required
      # to get it to 3. In the second trial we have a stone at 1 and 1 correct
      # potion and some potions which will take it away from the 3. It should
      # take the knowledge gained from trial 1 and apply it to trial 2.
      {'perceived_items': utils.EpisodeItems(
          stones=[[Stone(0, [-1, -1, -1])], [Stone(0, [1, 1, -1])]],
          potions=[[Potion(0, 0, 1), Potion(1, 1, 1), Potion(2, 2, 1)],
                   [Potion(0, 0, -1), Potion(1, 1, -1), Potion(2, 2, 1)]]),
       'potion_map': _ALL_FIXED_POTION_MAP,
       'stone_map': _ALL_FIXED_STONE_MAP,
       'graph': _ALL_FIXED_GRAPH,
       'expected_rewards': [15, 15]},
      {'perceived_items': utils.EpisodeItems(
          stones=[[Stone(0, [-1, -1, -1])], [Stone(0, [1, -1, 1])]],
          potions=[[Potion(0, 0, 1), Potion(1, 1, 1), Potion(2, 2, 1)],
                   [Potion(0, 0, -1), Potion(1, 1, -1), Potion(2, 2, 1)]]),
       'potion_map': _ALL_FIXED_POTION_MAP,
       'stone_map': _ALL_FIXED_STONE_MAP,
       'graph': _ALL_FIXED_GRAPH,
       'expected_rewards': [15, 1]},
      {'perceived_items': utils.EpisodeItems(
          stones=[
              [Stone(0, [-1, -1, -1])],
              [Stone(0, [1, -1, 1])],
              [Stone(0, [-1, -1, -1])],
              [Stone(0, [1, 1, 1])],
              [Stone(0, [1, 1, -1])],
              [Stone(0, [1, 1, -1])],
              [Stone(0, [1, -1, -1])],
              [Stone(0, [1, -1, -1])],
              [Stone(0, [1, -1, -1])],
              [Stone(0, [-1, -1, 1])]],
          potions=[
              [Potion(0, 0, -1), Potion(1, 1, -1), Potion(2, 0, 1),
               Potion(3, 2, -1), Potion(4, 0, 1), Potion(5, 0, -1)],
              [Potion(0, 0, 1), Potion(1, 0, -1), Potion(2, 0, -1),
               Potion(3, 0, -1), Potion(4, 0, 1), Potion(5, 2, -1)],
              [Potion(0, 0, -1), Potion(1, 1, 1), Potion(2, 2, -1),
               Potion(3, 1, -1), Potion(4, 2, -1), Potion(5, 2, -1)],
              [Potion(0, 1, -1), Potion(1, 0, -1), Potion(2, 1, -1),
               Potion(3, 1, -1), Potion(4, 0, -1), Potion(5, 0, -1)],
              [Potion(0, 1, 1), Potion(1, 0, -1), Potion(2, 1, -1),
               Potion(3, 2, 1), Potion(4, 1, 1), Potion(5, 0, -1)],
              [Potion(0, 2, -1), Potion(1, 1, 1), Potion(2, 2, 1),
               Potion(3, 0, -1), Potion(4, 0, -1), Potion(5, 0, 1)],
              [Potion(0, 0, 1), Potion(1, 0, 1), Potion(2, 0, -1),
               Potion(3, 2, -1), Potion(4, 2, -1), Potion(5, 1, 1)],
              [Potion(0, 2, -1), Potion(1, 1, 1), Potion(2, 1, -1),
               Potion(3, 0, 1), Potion(4, 2, -1), Potion(5, 2, -1)],
              [Potion(0, 2, 1), Potion(1, 2, -1), Potion(2, 2, -1),
               Potion(3, 2, 1), Potion(4, 1, -1), Potion(5, 1, -1)],
              [Potion(0, 1, -1), Potion(1, 0, -1), Potion(2, 2, 1),
               Potion(3, 2, 1), Potion(4, 2, -1), Potion(5, 1, 1)]]),
       'potion_map': _ALL_FIXED_POTION_MAP,
       'stone_map': _ALL_FIXED_STONE_MAP,
       'graph': _ALL_FIXED_GRAPH,
       'expected_rewards': [0, 0, 0, 15, 15, 15, 1, 1, 1, 1]},
      # This is a case that the oracle fails because it has to use a long path
      # instead of a short one.
      {'perceived_items': utils.EpisodeItems(
          stones=[[Stone(0, [-1, -1, -1]), Stone(1, [1, 1, -1])]],
          potions=[[Potion(0, 0, 1), Potion(1, 1, 1), Potion(2, 2, 1)]]),
       'potion_map': _ALL_FIXED_POTION_MAP,
       'stone_map': _ALL_FIXED_STONE_MAP,
       'graph': _BOTTLENECK1_GRAPH,
       'expected_rewards': [16],
       'bonus': 12,
       # Give it the exact constraint, stone map and potion map so it can run
       # as the oracle.
       'level_name': 'all_fixed_bottleneck1'},
      # This is a case where without sorting on search depth for actions of
      # equal reward, the search_oracle takes the longer path.
      {'perceived_items': utils.EpisodeItems(
          stones=[[Stone(0, [-1, -1, -1]), Stone(1, [-1, -1, 1])]],
          potions=[[Potion(0, 0, -1), Potion(1, 0, 1), Potion(2, 2, 1)]]),
       'potion_map': _ALL_FIXED_POTION_MAP,
       'stone_map': _ALL_FIXED_STONE_MAP,
       'graph': _BOTTLENECK1_GRAPH,
       'expected_rewards': [1],
       'bonus': 12,
       # Give it the exact constraint, stone map and potion map so it can run
       # as the oracle.
       'level_name': 'all_fixed_bottleneck1',
       # We expect the only events to be applying potion 1 to stone 1 and then
       # putting it into the cauldron.
       'expected_events': [ev_trk.OrderedEvents([
           # First put stone 1 into potion 1
           ev_trk.SingleEvent(1, {1}),
           # Then put stone 1 into the cauldron
           ev_trk.SingleEvent(1, {-1})
       ])],
       'non_events': [ev_trk.AnyOrderEvents({
           # Stone 0 should not be put into any potions or the cauldron.
           ev_trk.SingleEvent(0, {0, 1, 2, -1}),
           # Stone 1 should not be put into any other potions.
           ev_trk.SingleEvent(1, {0, 2}),
       })]},
      # Tests for the ideal explorer.
      # Ideal observer wouldn't use the potion as it minimises search depth if
      # it cannot get reward but ideal explorer will use it to minimise num
      # world states.
      {'minimise_world_states': False,
       'perceived_items': utils.EpisodeItems(
           stones=[[Stone(0, [-1, -1, -1])]],
           potions=[[Potion(0, 0, 1)]]),
       'potion_map': _ALL_FIXED_POTION_MAP,
       'stone_map': _ALL_FIXED_STONE_MAP,
       'graph': _BOTTLENECK1_GRAPH,
       'expected_rewards': [0],
       'bonus': 12,
       'non_events': [
           # Do not put stone 0 into potion 0 as we are minimising search depth.
           ev_trk.SingleEvent(0, {0}),
       ]},
      {'minimise_world_states': True,
       'perceived_items': utils.EpisodeItems(
           stones=[[Stone(0, [-1, -1, -1])]],
           potions=[[Potion(0, 0, 1)]]),
       'potion_map': _ALL_FIXED_POTION_MAP,
       'stone_map': _ALL_FIXED_STONE_MAP,
       'graph': _BOTTLENECK1_GRAPH,
       'expected_rewards': [0],
       'bonus': 12,
       'expected_events': [
           # Put stone 0 into potion 0 to reduce num world states.
           ev_trk.SingleEvent(0, {0}),
       ]},
  )
  def test_multiple_trials(
      self, perceived_items, potion_map, stone_map, graph, expected_rewards,
      bonus=12,
      level_name='perceptual_mapping_randomized_with_random_bottleneck',
      expected_events=None, non_events=None, minimise_world_states=False):

    precomputed = IdealObserverTest.level_name_to_precomputed[level_name]

    reward_weights = stones_and_potions.RewardWeights([1, 1, 1], 0, bonus)
    symbolic_bot_trackers_from_env = functools.partial(
        add_trackers_to_env, reward_weights=reward_weights,
        precomputed=precomputed, init_belief_state=None)
    results = symbolic_alchemy_bots.get_multi_trial_ideal_observer_reward(
        perceived_items,
        utils.Chemistry(potion_map, stone_map, graph, np.eye(3)),
        reward_weights, precomputed, minimise_world_states,
        symbolic_bot_trackers_from_env)
    per_trial = results['score']['per_trial']
    event_trackers = results['matrix_event']['event_tracker']

    self.assertLen(per_trial, len(expected_rewards))
    for trial_reward, expected_trial_reward in zip(per_trial, expected_rewards):
      self.assertEqual(trial_reward, expected_trial_reward, 4)

    if expected_events is not None:
      self.assertLen(event_trackers, len(expected_events))
      for event_tracker, expected_event in zip(event_trackers, expected_events):
        self.assertTrue(expected_event.occurs(event_tracker.events))
    if non_events is not None:
      self.assertLen(event_trackers, len(non_events))
      for event_tracker, non_event in zip(event_trackers, non_events):
        self.assertFalse(non_event.occurs(event_tracker.events))


if __name__ == '__main__':
  absltest.main()
