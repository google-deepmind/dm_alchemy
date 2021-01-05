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
"""Python utility functions for loading DeepMind Alchemy."""

import codecs
import json
import math
import os
import re
import subprocess
import time

from absl import logging
import dataclasses
from dm_alchemy import partial_array_specs
import dm_env
from dm_env import specs as array_specs
import docker
import grpc
import numpy as np
import portpicker
import tree

from dm_alchemy.protos import events_pb2
from dm_env_rpc.v1 import connection as dm_env_rpc_connection
from dm_env_rpc.v1 import dm_env_adaptor
from dm_env_rpc.v1 import dm_env_rpc_pb2
from dm_env_rpc.v1 import error
from dm_env_rpc.v1 import tensor_utils

# Maximum number of times to attempt gRPC connection.
_MAX_CONNECTION_ATTEMPTS = 10

# Port to expect the docker environment to internally listen on.
_DOCKER_INTERNAL_GRPC_PORT = 10000

_DEFAULT_DOCKER_IMAGE_NAME = 'gcr.io/deepmind-environments/alchemy:v1.0.0'

_ALCHEMY_OBSERVATIONS = ('RGB_INTERLEAVED', 'ACCELERATION', 'HAND_FORCE',
                         'HAND_IS_HOLDING', 'HAND_DISTANCE', 'Score', 'events')

ALCHEMY_LEVEL_NAMES = frozenset((
    'alchemy/perceptual_mapping_randomized_with_rotation_and_random_bottleneck',
    'alchemy/all_fixed',
    'alchemy/all_fixed_w_shaping',
    'alchemy/perceptual_mapping_randomized_with_random_bottleneck',
) + tuple(f'alchemy/evaluation_episodes/{i}' for i in range(1000)))


@dataclasses.dataclass
class _ConnectionDetails:
  channel: grpc.Channel
  connection: dm_env_rpc_connection.Connection
  specs: dm_env_rpc_pb2.ActionObservationSpecs


def _maybe_as_partial_spec(spec: array_specs.Array):
  if -1 not in spec.shape:
    return spec

  if isinstance(spec, array_specs.BoundedArray):
    raise ValueError('Partial bounded arrays are not yet handled.')

  return partial_array_specs.PartialArray(spec.shape, spec.dtype, spec.name)


def _unpack_world_event(event):
  decoded = events_pb2.WorldEvent()
  if not event.Unpack(decoded):
    raise ValueError('Event could not be decoded to WorldEvent. {event}'.format(
        event=str(event)))
  return decoded


class _AlchemyEnv(dm_env_adaptor.DmEnvAdaptor):
  """An implementation of dm_env_rpc.DmEnvAdaptor for Alchemy env."""

  def __init__(self, connection_details, requested_observations,
               num_action_repeats):
    super().__init__(connection_details.connection,
                     connection_details.specs, requested_observations)
    self._channel = connection_details.channel
    self._num_action_repeats = num_action_repeats
    self._events = []

  def close(self):
    super().close()
    self._channel.close()

  def step(self, action):
    """Implementation of dm_env.step that supports repeated actions."""

    discount = None
    reward = None
    self._events = []
    for _ in range(self._num_action_repeats):
      next_timestep = super().step(action)

      # Accumulate reward per timestep.
      if next_timestep.reward is not None:
        reward = (reward or 0.) + next_timestep.reward

      # Calculate the product for discount.
      if next_timestep.discount is not None:
        discount = discount if discount else []
        discount.append(next_timestep.discount)

      timestep = dm_env.TimeStep(next_timestep.step_type, reward,
                                 # Note: np.product(None) returns None.
                                 np.product(discount),
                                 next_timestep.observation)
      self._events.extend([_unpack_world_event(event)
                           for event in timestep.observation['events']])

      if timestep.last():
        return timestep

    return timestep

  def observation_spec(self):
    return tree.map_structure(
        _maybe_as_partial_spec, super().observation_spec())

  def events(self):
    return self._events


class _AlchemyContainerEnv(_AlchemyEnv):
  """An implementation of _AlchemyEnv.

    Ensures that the provided Docker container is closed on exit.
  """

  def __init__(self, container, **base_kwargs):
    super().__init__(**base_kwargs)
    self._container = container

  def close(self):
    super().close()
    try:
      self._container.kill()
    except docker.errors.NotFound:
      pass  # Ignore, container has already been closed.


class _AlchemyProcessEnv(_AlchemyEnv):
  """An implementation of _AlchemyEnv.

    Ensures that the provided running process is closed on exit.
  """

  def __init__(self, connection_details, requested_observations,
               num_action_repeats, process):
    super().__init__(connection_details, requested_observations,
                     num_action_repeats)
    self._process = process

  def close(self):
    super().close()
    self._process.terminate()
    self._process.wait()


def _check_grpc_channel_ready(channel):
  """Helper function to check the gRPC channel is ready N times."""
  for _ in range(_MAX_CONNECTION_ATTEMPTS - 1):
    try:
      return grpc.channel_ready_future(channel).result(timeout=1)
    except grpc.FutureTimeoutError:
      pass
  return grpc.channel_ready_future(channel).result(timeout=1)


def _can_send_message(connection):
  """Returns if `connection` is healthy and able to process requests."""
  try:
    # This should return a response with an error unless the server isn't yet
    # receiving requests.
    connection.send(dm_env_rpc_pb2.StepRequest())
  except error.DmEnvRpcError:
    return True
  except grpc.RpcError:
    return False
  return True


def _create_channel_and_connection(port):
  """Returns a tuple of `(channel, connection)`."""
  for i in range(_MAX_CONNECTION_ATTEMPTS):
    channel = grpc.secure_channel('localhost:{}'.format(port),
                                  grpc.local_channel_credentials())
    _check_grpc_channel_ready(channel)
    connection = dm_env_rpc_connection.Connection(channel)
    if _can_send_message(connection):
      break
    else:
      # A gRPC server running within Docker sometimes reports that the channel
      # is ready but transitively returns an error (status code 14) on first
      # use.  Giving the server some time to breath and retrying often fixes the
      # problem.
      connection.close()
      channel.close()
      time.sleep(math.pow(1.4, i))

  return channel, connection


def _parse_exception_message(message):
  """Returns a human-readable version of a dm_env_rpc json error message."""
  try:
    match = re.match(r'^message\:\ \"(.*)\"$', message)
    json_data = codecs.decode(match.group(1), 'unicode-escape')
    parsed_json_data = json.loads(json_data)
    return ValueError(json.dumps(parsed_json_data, indent=4))
  except:  # pylint: disable=bare-except
    return message


def _wrap_send(send):
  """Wraps `send` in order to reformat exceptions."""
  try:
    return send()
  except ValueError as e:
    e.args = [_parse_exception_message(e.args[0])]
    raise


def _connect_to_environment(port, settings):
  """Helper function for connecting to a running Alchemy environment."""
  if settings.level_name not in ALCHEMY_LEVEL_NAMES:
    raise ValueError(
        'Level named "{}" is not a valid dm_alchemy level.'.format(
            settings.level_name))
  channel, connection = _create_channel_and_connection(port)
  original_send = connection.send
  connection.send = lambda request: _wrap_send(lambda: original_send(request))
  world_name = connection.send(
      dm_env_rpc_pb2.CreateWorldRequest(
          settings={
              'seed': tensor_utils.pack_tensor(settings.seed),
              'episodeId': tensor_utils.pack_tensor(0),
              'levelName': tensor_utils.pack_tensor(settings.level_name),
              'EventSubscriptionRegex': tensor_utils.pack_tensor('DeepMind/.*'),
          })).world_name
  join_world_settings = {
      'width': tensor_utils.pack_tensor(settings.width),
      'height': tensor_utils.pack_tensor(settings.height),
  }
  specs = connection.send(
      dm_env_rpc_pb2.JoinWorldRequest(
          world_name=world_name, settings=join_world_settings)).specs
  return _ConnectionDetails(channel=channel, connection=connection, specs=specs)


@dataclasses.dataclass
class EnvironmentSettings:
  """Collection of settings used to start a specific Alchemy level.

    Required attributes:
      seed: Seed to initialize the environment's RNG.
      level_name: Name of the level to load.
    Optional attributes:
      width: Width (in pixels) of the desired RGB observation; defaults to 96.
      height: Height (in pixels) of the desired RGB observation; defaults to 72.
      num_action_repeats: Number of times to step the environment with the
        provided action in calls to `step()`.
  """
  seed: int
  level_name: str
  width: int = 96
  height: int = 72
  num_action_repeats: int = 1


def _validate_environment_settings(settings):
  """Helper function to validate the provided environment settings."""
  if settings.num_action_repeats <= 0:
    raise ValueError('num_action_repeats must have a positive value.')
  if settings.width <= 0 or settings.height <= 0:
    raise ValueError('width and height must have a positive value.')


def load_from_disk(path, settings, environment_variables=None):
  """Loads Alchemy from disk.

  Args:
    path: Directory containing dm_alchemy environment.
    settings: EnvironmentSettings required to start the environment.
    environment_variables: Optional dictionary containing the system environment
      variables to be set for the new process. The dictionary maps variable
      names to their string values.

  Returns:
    An implementation of dm_env.Environment.

  Raises:
    RuntimeError: If unable to start environment process.
  """
  environment_variables = environment_variables or {}

  _validate_environment_settings(settings)

  executable_path = os.path.join(path, 'Linux64Player')
  libosmesa_path = os.path.join(path, 'external_libosmesa_llvmpipe.so')
  if not os.path.exists(executable_path) or not os.path.exists(libosmesa_path):
    raise RuntimeError(
        'Cannot find dm_alchemy executable or dependent files at path: {}'
        .format(path))

  port = portpicker.pick_unused_port()

  process_flags = [
      executable_path,
      # Unity command-line flags.
      '-logfile',
      '-batchmode',
      '-noaudio',
      # Other command-line flags.
      '--logtostderr',
      '--server_type=GRPC',
      '--uri_address=[::]:{}'.format(port),
  ]

  os.environ.update({
      'UNITY_RENDERER': 'software',
      'UNITY_OSMESA_PATH': libosmesa_path,
  })
  os.environ.update(environment_variables)

  process = subprocess.Popen(
      process_flags, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
  if process.poll() is not None:
    raise RuntimeError('Failed to start dm_alchemy process correctly.')

  return _AlchemyProcessEnv(
      _connect_to_environment(port, settings), _ALCHEMY_OBSERVATIONS,
      settings.num_action_repeats, process)


def load_from_docker(settings, environment_variables=None, name=None):
  """Loads Alchemy env from docker container.

  Args:
    settings: EnvironmentSettings required to start the environment.
    environment_variables: Optional dictionary containing the system environment
      variables to be set inside the container. The dictionary maps variable
      names to their string values.
    name: Optional name of Docker image that contains the Alchemy
      environment. If left unset, uses the default name.

  Returns:
    An implementation of dm_env.Environment
  """
  _validate_environment_settings(settings)

  environment_variables = environment_variables or {}
  name = name or _DEFAULT_DOCKER_IMAGE_NAME
  client = docker.from_env()

  port = portpicker.pick_unused_port()

  try:
    client.images.get(name)
  except docker.errors.ImageNotFound:
    logging.info('Downloading docker image "%s"...', name)
    client.images.pull(name)
    logging.info('Download finished.')

  container = client.containers.run(
      name,
      auto_remove=True,
      detach=True,
      ports={_DOCKER_INTERNAL_GRPC_PORT: port},
      environment=environment_variables)

  kwargs = dict(
      connection_details=_connect_to_environment(port, settings),
      requested_observations=_ALCHEMY_OBSERVATIONS,
      num_action_repeats=settings.num_action_repeats,
      container=container)
  return _AlchemyContainerEnv(**kwargs)
