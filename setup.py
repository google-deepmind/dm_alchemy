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
"""Install script for setuptools."""

from distutils import cmd
import imp
import os

import pkg_resources
from setuptools import find_namespace_packages
from setuptools import setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py


_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Tuple of proto message definitions to build Python bindings for. Paths must
# be relative to root directory.
_DM_ALCHEMY_PROTOS = (
    'dm_alchemy/protos/alchemy.proto',
    'dm_alchemy/protos/trial.proto',
    'dm_alchemy/protos/color_info.proto',
    'dm_alchemy/protos/unity_types.proto',
    'dm_alchemy/protos/episode_info.proto',
    'dm_alchemy/protos/events.proto',
    'dm_alchemy/protos/hypercube.proto',
    'dm_alchemy/encode/chemistries.proto',
    'dm_alchemy/encode/symbolic_actions.proto',
    'dm_alchemy/encode/precomputed_maps.proto')


class _GenerateProtoFiles(cmd.Command):
  """Command to generate protobuf bindings for dm_alchemy."""

  descriptions = 'Generates Python protobuf bindings.'
  user_options = []

  def initialize_options(self):
    pass

  def finalize_options(self):
    pass

  def run(self):
    # Import grpc_tools here, after setuptools has installed setup_requires
    # dependencies.
    from grpc_tools import protoc  # pylint: disable=g-import-not-at-top

    grpc_protos_include = pkg_resources.resource_filename(
        'grpc_tools', '_proto')

    for proto_path in _DM_ALCHEMY_PROTOS:
      proto_args = [
          'grpc_tools.protoc',
          '--proto_path={}'.format(grpc_protos_include),
          '--proto_path={}'.format(_ROOT_DIR),
          '--python_out={}'.format(_ROOT_DIR),
          '--grpc_python_out={}'.format(_ROOT_DIR),
          os.path.join(_ROOT_DIR, proto_path),
      ]
      if protoc.main(proto_args) != 0:
        raise RuntimeError('ERROR: {}'.format(proto_args))


class _BuildExt(build_ext):
  """Generate protobuf bindings in build_ext stage."""

  def run(self):
    self.run_command('generate_protos')
    build_ext.run(self)


class _BuildPy(build_py):
  """Generate protobuf bindings in build_py stage."""

  def run(self):
    self.run_command('generate_protos')
    build_py.run(self)

setup(
    name='dm-alchemy',
    version=imp.load_source('_version',
                            'dm_alchemy/_version.py').__version__,
    description=('DeepMind Alchemy environment, a meta-reinforcement learning'
                 'benchmark environment for deep RL agents.'),
    author='DeepMind',
    license='Apache License, Version 2.0',
    keywords='reinforcement-learning python machine learning',
    packages=find_namespace_packages(exclude=['examples']),
    package_data={
        'dm_alchemy.encode': ['*.proto'],
        'dm_alchemy.protos': ['*.proto'],
        'dm_alchemy.chemistries': ['**/**'],
        'dm_alchemy.ideal_observer.data': ['**/**'],
        'dm_alchemy.agent_events': ['**'],
    },
    install_requires=[
        'absl-py',
        'dataclasses',
        'dm-env',
        'dm-env-rpc>=1.0.4',
        'dm-tree',
        'docker',
        'grpcio',
        'numpy',
        'scipy>=1.4.0',
        'portpicker',
    ],
    tests_require=['nose'],
    python_requires='>=3.6.1',
    setup_requires=['grpcio-tools'],
    extras_require={
        'examples': [
            'pygame',
            'ipykernel',
            'matplotlib',
            'seaborn',
        ]},
    cmdclass={
        'build_ext': _BuildExt,
        'build_py': _BuildPy,
        'generate_protos': _GenerateProtoFiles,
    },
    test_suite='nose.collector',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
