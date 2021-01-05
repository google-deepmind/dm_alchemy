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
"""Provide meta data about symbolic alchemy levels."""

# Each latent dimension contributes equal value to the stone reward.
_VALUE_COEFFICIENTS = [1, 1, 1]
_VALUE_OFFSET = 0
# Getting the best stone increases the value by 12.
_BONUS = 12


def to_meta_data(level_name: str):
  potion_reward = 1 if 'shaping' in level_name else 0
  vary_spawns = 'vary_spawns' in level_name
  return _VALUE_COEFFICIENTS, _VALUE_OFFSET, potion_reward, _BONUS, vary_spawns
