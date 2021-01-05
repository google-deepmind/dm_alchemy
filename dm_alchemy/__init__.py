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
"""Python utilities for running dm_alchemy."""

from dm_alchemy import _load_environment
from dm_alchemy._version import __version__

EnvironmentSettings = _load_environment.EnvironmentSettings

LEVEL_NAMES = _load_environment.ALCHEMY_LEVEL_NAMES

load_from_disk = _load_environment.load_from_disk
load_from_docker = _load_environment.load_from_docker
