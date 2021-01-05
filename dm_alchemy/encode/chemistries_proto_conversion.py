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
"""Functions for converting to and from chemistry protos."""

from typing import List, Sequence, Tuple

from dm_alchemy import io
from dm_alchemy.encode import chemistries_pb2
from dm_alchemy.encode import precomputed_maps_proto_conversion
from dm_alchemy.ideal_observer import precomputed_maps
from dm_alchemy.types import stones_and_potions
from dm_alchemy.types import utils


def chemistry_to_proto(
    chemistry: utils.Chemistry,
) -> chemistries_pb2.Chemistry:
  perm_index_to_index, _ = precomputed_maps.get_perm_index_conversion()
  return chemistries_pb2.Chemistry(
      potion_map=chemistry.potion_map.index(perm_index_to_index),
      stone_map=chemistry.stone_map.index(),
      graph=precomputed_maps_proto_conversion.graph_to_proto(chemistry.graph),
      rotation=stones_and_potions.rotation_to_angles(chemistry.rotation))


def proto_to_chemistry(
    proto: chemistries_pb2.Chemistry
) -> utils.Chemistry:
  _, index_to_perm_index = precomputed_maps.get_perm_index_conversion()
  return utils.Chemistry(
      potion_map=stones_and_potions.potion_map_from_index(
          stones_and_potions.PotionMapIndex(proto.potion_map),
          index_to_perm_index),
      stone_map=stones_and_potions.stone_map_from_index(
          stones_and_potions.StoneMapIndex(proto.stone_map)),
      graph=precomputed_maps_proto_conversion.proto_to_graph(proto.graph),
      rotation=stones_and_potions.rotation_from_angles(proto.rotation))


def trial_items_to_proto(
    trial_items: utils.TrialItems
) -> chemistries_pb2.TrialItems:
  return chemistries_pb2.TrialItems(
      stones=[s.latent_stone().index() for s in trial_items.stones],
      potions=[p.latent_potion().index() for p in trial_items.potions])


def proto_to_trial_items(
    proto: chemistries_pb2.TrialItems
) -> utils.TrialItems:
  return utils.TrialItems(
      stones=[stones_and_potions.latent_stone_from_index(
          stones_and_potions.LatentStoneIndex(s)) for s in proto.stones],
      potions=[stones_and_potions.latent_potion_from_index(
          stones_and_potions.LatentPotionIndex(p)) for p in proto.potions])


def episode_items_to_proto(
    episode_items: utils.EpisodeItems
) -> chemistries_pb2.EpisodeItems:
  return chemistries_pb2.EpisodeItems(
      trial_items=[trial_items_to_proto(t) for t in episode_items.trials])


def proto_to_episode_items(
    proto: chemistries_pb2.EpisodeItems
) -> utils.EpisodeItems:
  items = utils.EpisodeItems(stones=[], potions=[])
  items.trials = [proto_to_trial_items(p) for p in proto.trial_items]
  return items


def chemistry_and_items_to_proto(
    chemistry: utils.Chemistry,
    episode_items: utils.EpisodeItems,
) -> chemistries_pb2.ChemistryAndItems:
  return chemistries_pb2.ChemistryAndItems(
      chemistry=chemistry_to_proto(chemistry),
      items=episode_items_to_proto(episode_items))


def proto_to_chemistry_and_items(
    proto: chemistries_pb2.ChemistryAndItems
) -> Tuple[utils.Chemistry, utils.EpisodeItems]:
  return (proto_to_chemistry(proto.chemistry),
          proto_to_episode_items(proto.items))


def chemistries_and_items_to_proto(
    chemistries_and_items: Sequence[Tuple[utils.Chemistry, utils.EpisodeItems]],
) -> chemistries_pb2.ChemistriesAndItems:
  return chemistries_pb2.ChemistriesAndItems(
      chemistries=[chemistry_and_items_to_proto(chemistry, episode_items)
                   for chemistry, episode_items in chemistries_and_items])


def proto_to_chemistries_and_items(
    proto: chemistries_pb2.ChemistriesAndItems
) -> List[Tuple[utils.Chemistry, utils.EpisodeItems]]:
  return [(proto_to_chemistry(chem.chemistry),
           proto_to_episode_items(chem.items)) for chem in proto.chemistries]


def write_chemistries_and_items(
    chemistries_and_items: Sequence[Tuple[utils.Chemistry, utils.EpisodeItems]],
    filename: str
) -> None:
  proto = chemistries_and_items_to_proto(chemistries_and_items)
  io.write_proto(filename, proto.SerializeToString())


def load_chemistries_and_items(
    filename: str
) -> List[Tuple[utils.Chemistry, utils.EpisodeItems]]:
  serialized = io.read_proto(filename)
  proto = chemistries_pb2.ChemistriesAndItems.FromString(serialized)
  return proto_to_chemistries_and_items(proto)
