# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
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
"""Example random agent for interacting with DeepMind Alchemy environment."""

from absl import app
from absl import flags
from absl import logging
import dm_alchemy
from dm_env import specs
import numpy as np

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'docker_image_name', None,
    'Name of the Docker image that contains the Alchemy environment. '
    'If None, uses the default docker image name.')

flags.DEFINE_integer('seed', 123, 'Environment seed.')
flags.DEFINE_string(
    'level_name',
    'alchemy/perceptual_mapping_randomized_with_rotation_and_random_bottleneck',
    'Name of Alchemy task to run.')


class RandomAgent:
  """Basic random agent for DeepMind Alchemy environment."""

  def __init__(self, action_spec):
    self.action_spec = action_spec

  def act(self):
    action = {}

    for name, spec in self.action_spec.items():
      # Uniformly sample BoundedArray actions.
      if isinstance(spec, specs.BoundedArray):
        action[name] = np.random.uniform(spec.minimum, spec.maximum, spec.shape)
      else:
        action[name] = spec.generate_value()
    return action


def main(_):
  env_settings = dm_alchemy.EnvironmentSettings(
      seed=FLAGS.seed, level_name=FLAGS.level_name)
  with dm_alchemy.load_from_docker(
      name=FLAGS.docker_image_name, settings=env_settings) as env:
    agent = RandomAgent(env.action_spec())

    timestep = env.reset()
    score = 0
    while not timestep.last():
      action = agent.act()
      timestep = env.step(action)

      if timestep.reward:
        score += timestep.reward
        logging.info('Total score: %1.1f, reward: %1.1f', score,
                     timestep.reward)


if __name__ == '__main__':
  app.run(main)
