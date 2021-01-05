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
"""Tests for the symbolic alchemy wrapper."""

from absl.testing import absltest
from dm_alchemy import symbolic_alchemy_wrapper
from dm_alchemy.protos import alchemy_pb2
from dm_alchemy.types import graphs
from dm_alchemy.types import stones_and_potions
from dm_alchemy.types import unity_python_conversion
from dm_alchemy.types import utils
import dm_env
import numpy as np

from google.protobuf import any_pb2
from dm_alchemy.protos import events_pb2
from dm_alchemy.protos import trial_pb2

Stone = stones_and_potions.Stone
Potion = stones_and_potions.Potion
LatentStone = stones_and_potions.LatentStone
LatentPotion = stones_and_potions.LatentPotion
AlignedStone = stones_and_potions.AlignedStone
PerceivedPotion = stones_and_potions.PerceivedPotion
CAULDRON = stones_and_potions.CAULDRON


def encode_event(name, event):
  any_proto = any_pb2.Any()
  any_proto.Pack(event)
  world_event = events_pb2.WorldEvent(
      name=name + ':deepmind.dmworlds.WorldEvent', detail=any_proto)
  return world_event


class Mock3DEnv:
  """A mock 3d environment which we can set events on."""

  def __init__(self):
    self._new_trial = False
    self._next_step_new_trial = False
    self._last_step = True
    self.chemistry = None
    self.items = None
    self._trial_number = -1
    self._used_events = []
    self._next_step_used_events = []

  def set_chemistry_and_items(self, chemistry, items):
    assert isinstance(chemistry, utils.Chemistry)
    self.chemistry = chemistry
    self.items = items

  def set_new_trial(self):
    self._next_step_new_trial = True
    self._trial_number += 1

  def set_potion_used(self, potion_instance_id, stone_instance_id):
    self._next_step_used_events.append(
        ('DeepMind/Alchemy/PotionUsed', alchemy_pb2.PotionUsed(
            potion_instance_id=potion_instance_id,
            stone_instance_id=stone_instance_id)))

  def set_stone_used(self, stone_instance_id):
    self._next_step_used_events.append(
        ('DeepMind/Alchemy/StoneUsed', alchemy_pb2.StoneUsed(
            stone_instance_id=stone_instance_id)))

  def events(self):
    events = []
    if self._new_trial:
      if self._trial_number == 0:
        unity_chemistry, rotation_mapping = (
            unity_python_conversion.to_unity_chemistry(
                self.chemistry))
        events.append(
            ('DeepMind/Alchemy/ChemistryCreated', alchemy_pb2.ChemistryCreated(
                chemistry=unity_chemistry, rotation_mapping=rotation_mapping)))
      else:
        events.append(
            ('DeepMind/Trial/TrialEnded', trial_pb2.TrialEnded(
                trial_id=self._trial_number - 1)))
      events.append(
          ('DeepMind/Alchemy/CauldronCreated', alchemy_pb2.CauldronCreated()))
      for potion in self.items.trials[self._trial_number].potions:
        latent_potion = potion.latent_potion()
        perceived_potion = self.chemistry.potion_map.apply_inverse(
            latent_potion)
        potion_properties = unity_python_conversion.to_potion_unity_properties(
            perceived_potion=perceived_potion, latent_potion=latent_potion,
            graph=self.chemistry.graph)
        events.append(
            ('DeepMind/Alchemy/PotionCreated', alchemy_pb2.PotionCreated(
                potion_instance_id=potion.idx,
                potion_properties=potion_properties)))
      for stone in self.items.trials[self._trial_number].stones:
        latent_stone = stone.latent_stone()
        aligned_stone = self.chemistry.stone_map.apply_inverse(latent_stone)
        perceived_stone = stones_and_potions.unalign(
            aligned_stone, self.chemistry.rotation)
        stone_properties = unity_python_conversion.to_stone_unity_properties(
            perceived_stone=perceived_stone, latent_stone=latent_stone)
        events.append(
            ('DeepMind/Alchemy/StoneCreated', alchemy_pb2.StoneCreated(
                stone_instance_id=stone.idx,
                stone_properties=stone_properties)))
      events.append(
          ('DeepMind/Trial/TrialStarted', trial_pb2.TrialStarted(
              trial_id=self._trial_number)))
    events.extend(self._used_events)
    return [encode_event(name, event) for name, event in events]

  def _timestep(self):
    # The timestep doesn't matter.
    return dm_env.TimeStep(None, None, None, None)

  def step(self, unused_action):
    del unused_action
    self._used_events = self._next_step_used_events
    self._next_step_used_events = []
    self._new_trial = self._next_step_new_trial
    self._next_step_new_trial = False
    if self._last_step:
      return self.reset()
    return self._timestep()

  def reset(self):
    self._last_step = False
    self.set_new_trial()
    return self._timestep()


class SymbolicAlchemyWrapperTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.env3d_mock = Mock3DEnv()
    self.chemistry = utils.Chemistry(
        potion_map=stones_and_potions.all_fixed_potion_map(),
        stone_map=stones_and_potions.all_fixed_stone_map(),
        graph=graphs.create_graph_from_constraint(
            graphs.no_bottleneck_constraints()[0]),
        rotation=np.eye(3))
    self.items = utils.EpisodeItems(
        potions=[[Potion(0, 0, 1)], [Potion(1, 2, -1)]],
        stones=[[Stone(2, [-1, -1, -1])], [Stone(3, [1, 1, 1])]])
    self.env3d_mock.set_chemistry_and_items(self.chemistry, self.items)
    self.wrapper = symbolic_alchemy_wrapper.SymbolicAlchemyWrapper(
        self.env3d_mock,
        'alchemy/perceptual_mapping_randomized_with_random_bottleneck')

  def test_items_generated_each_trial(self):
    # Once the trial has started the items in the symbolic environment should
    # match the ones we let the mock 3d environment generate.
    self.wrapper.reset()
    # Action is not important
    self.wrapper.step(action=None)
    self.assertEqual(
        self.wrapper.env_symbolic._chemistry.potion_map,
        self.chemistry.potion_map)
    self.assertEqual(
        self.wrapper.env_symbolic._chemistry.stone_map,
        self.chemistry.stone_map)
    self.assertEqual(
        graphs.constraint_from_graph(
            self.wrapper.env_symbolic._chemistry.graph),
        graphs.constraint_from_graph(self.chemistry.graph))
    self.assertEqual(
        self.wrapper.env_symbolic.game_state.existing_items(),
        self.items.trials[0])

    # Check that the items in the second trial are also correct.
    for trial in self.items.trials[1:]:
      self.env3d_mock.set_new_trial()
      self.wrapper.step(action=None)
      self.assertEqual(
          self.wrapper.env_symbolic.game_state.existing_items(),
          trial)

  def test_potion_used(self):
    # Reset and take a step to ensure the items are generated.
    self.wrapper.reset()
    self.wrapper.step(None)
    self.assertIsNotNone(self.wrapper.env_symbolic.game_state)
    self.assertNotEmpty(self.wrapper.env_symbolic.game_state.existing_potions())
    # Now ensure that on the next step a potion is used.
    self.env3d_mock.set_potion_used(potion_instance_id=0, stone_instance_id=2)
    self.wrapper.step(None)
    # Now the potion should be gone and the stone should have changed.
    self.assertEmpty(self.wrapper.env_symbolic.game_state.existing_potions())
    self.assertEqual(
        self.wrapper.env_symbolic.game_state.existing_stones(),
        [Stone(2, [1, -1, -1])])

  def test_stone_used(self):
    # Reset and take a step to ensure the items are generated.
    self.wrapper.reset()
    self.wrapper.step(None)
    # Now ensure that on the next step a stone is used.
    self.env3d_mock.set_stone_used(stone_instance_id=2)
    self.wrapper.step(None)
    # Now the stone should be gone.
    self.assertEmpty(self.wrapper.env_symbolic.game_state.existing_stones())


if __name__ == '__main__':
  absltest.main()
