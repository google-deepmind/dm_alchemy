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
"""Unpack events from the environment."""

import copy
from typing import List, Sequence, Tuple

from dm_alchemy.protos import alchemy_pb2
from dm_alchemy.types import stones_and_potions

from dm_alchemy.types import unity_python_conversion
from dm_alchemy.protos import events_pb2


def unpack_chemistry_and_rotation(
    event: events_pb2.WorldEvent
) -> Tuple[alchemy_pb2.Chemistry, alchemy_pb2.RotationMapping]:
  chem_created = alchemy_pb2.ChemistryCreated()
  event.detail.Unpack(chem_created)
  return chem_created.chemistry, chem_created.rotation_mapping


def unpack_potion_used(event: events_pb2.WorldEvent) -> Tuple[int, int]:
  potion_used = alchemy_pb2.PotionUsed()
  event.detail.Unpack(potion_used)
  return potion_used.potion_instance_id, potion_used.stone_instance_id


def unpack_stone_used(event: events_pb2.WorldEvent) -> int:
  stone_used = alchemy_pb2.StoneUsed()
  event.detail.Unpack(stone_used)
  return stone_used.stone_instance_id


def potions_used_on_step(
    events: Sequence[events_pb2.WorldEvent]
) -> List[Tuple[int, int]]:
  return [unpack_potion_used(event) for event in events
          if 'Alchemy/PotionUsed' in event.name]


def stones_used_on_step(events: Sequence[events_pb2.WorldEvent]) -> List[int]:
  return [unpack_stone_used(event) for event in events
          if 'Alchemy/StoneUsed' in event.name]


def get_stones(
    creation_events: Sequence[events_pb2.WorldEvent]
) -> List[Tuple[stones_and_potions.PerceivedStone, int]]:
  """Gets a list of Stone objects from creation events."""
  stones = []
  for event in creation_events:
    if 'StoneCreated' in event.name:
      stone_event = alchemy_pb2.StoneCreated()
      event.detail.Unpack(stone_event)
      perceived_stone = unity_python_conversion.unity_to_perceived_stone(
          stone_event.stone_properties)
      stones.append((perceived_stone, stone_event.stone_instance_id))
  return stones


def get_potions(
    creation_events: Sequence[events_pb2.WorldEvent]
) -> List[Tuple[stones_and_potions.PerceivedPotion, int]]:
  """Gets a list of Potion objects from creation events."""
  potions = []
  for event in creation_events:
    if 'PotionCreated' in event.name:
      potion_event = alchemy_pb2.PotionCreated()
      event.detail.Unpack(potion_event)
      perceived_potion = unity_python_conversion.unity_to_perceived_potion(
          potion_event.potion_properties)
      potions.append((perceived_potion, potion_event.potion_instance_id))
  return potions


def get_bottlenecks_and_rotation(
    creation_events: Sequence[events_pb2.WorldEvent]
) -> Tuple[alchemy_pb2.Chemistry, alchemy_pb2.RotationMapping]:
  """Gets the chemistry constraints from creation_events."""
  # search through trajectory for ChemistryCreated event
  chemistry_events = []
  for event in creation_events:
    if 'ChemistryCreated' in event.name:
      chem_event = alchemy_pb2.ChemistryCreated()
      event.detail.Unpack(chem_event)
      chemistry_events.append(chem_event)
  assert chemistry_events, 'Chemistry not found'
  return chemistry_events[0].chemistry, chemistry_events[0].rotation_mapping


def events_per_trial(
    trajectory: Sequence[Sequence[events_pb2.WorldEvent]]
) -> List[List[events_pb2.WorldEvent]]:
  """Split the events for the trajectory by trial."""
  per_trial_events = []
  trial_events = []
  for step in trajectory:
    for event in step:
      if 'TrialEnded' in event.name:
        trial_events = []
      elif 'TrialStarted' in event.name:
        per_trial_events.append(copy.deepcopy(trial_events))
      else:
        trial_events.append(event)
  return per_trial_events
