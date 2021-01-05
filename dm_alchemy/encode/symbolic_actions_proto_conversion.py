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
"""Functions for converting to and from sets of symbolic actions protos."""

from typing import List, Sequence

from dm_alchemy import event_tracker
from dm_alchemy.encode import precomputed_maps_pb2
from dm_alchemy.encode import symbolic_actions_pb2
import numpy as np


def trial_events_to_proto(
    trial_events: event_tracker.MatrixEventTracker,
) -> precomputed_maps_pb2.IntArray:
  return precomputed_maps_pb2.IntArray(
      shape=trial_events.events.shape,
      entries=trial_events.events.ravel().tolist())


def proto_to_trial_events(
    proto: precomputed_maps_pb2.IntArray,
) -> event_tracker.MatrixEventTracker:
  trial_tracker = event_tracker.MatrixEventTracker(
      num_stones=proto.shape[0], num_potions=proto.shape[1] - 1)
  trial_tracker.events = np.array(proto.entries, dtype=np.int).reshape(
      proto.shape)
  return trial_tracker


def episode_events_to_proto(
    episode_events: Sequence[event_tracker.MatrixEventTracker]
) -> symbolic_actions_pb2.EpisodeEvents:
  return symbolic_actions_pb2.EpisodeEvents(
      trial_events=[trial_events_to_proto(trial_events)
                    for trial_events in episode_events])


def proto_to_episode_events(
    proto: symbolic_actions_pb2.EpisodeEvents,
) -> List[event_tracker.MatrixEventTracker]:
  return [proto_to_trial_events(trial_events_proto)
          for trial_events_proto in proto.trial_events]


def evaluation_set_events_to_proto(
    evaluation_set: Sequence[Sequence[event_tracker.MatrixEventTracker]]
) -> symbolic_actions_pb2.EvaluationSetEvents:
  return symbolic_actions_pb2.EvaluationSetEvents(
      episode_events=[episode_events_to_proto(episode_events)
                      for episode_events in evaluation_set])


def proto_to_evaluation_set_events(
    proto: symbolic_actions_pb2.EvaluationSetEvents,
) -> List[List[event_tracker.MatrixEventTracker]]:
  return [proto_to_episode_events(episode_events_proto)
          for episode_events_proto in proto.episode_events]
