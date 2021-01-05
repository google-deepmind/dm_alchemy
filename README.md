# `dm_alchemy`: DeepMind Alchemy environment

**[Overview](#overview)** | **[Requirements](#requirements)** |
**[Installation](#installation)** | **[Usage](#usage)** | **[Documentation]** |
**[Tutorial]** | **[Paper]** | **[Blog post]**

The *DeepMind Alchemy environment* is a meta-reinforcement learning benchmark
that presents tasks sampled from a task distribution with deep underlying
structure. It was created to test for the ability of agents to reason and plan
via latent state inference, as well as useful exploration and experimentation.
It is [Unity-based](http://unity3d.com/).

<p align="center">
    <img src="docs/gameplay.gif" align="center" width="500">
</p>

## Overview

This environment is provided through pre-packaged
[Docker containers](http://www.docker.com).

This package consists of support code to run these Docker containers. You
interact with the task environment via a
[`dm_env`](http://www.github.com/deepmind/dm_env) Python interface.

Please see the [documentation](docs/index.md) for more detailed information on
the available tasks, actions and observations.

## Requirements

`dm_alchemy` requires [Docker](https://www.docker.com),
[Python](https://www.python.org/) 3.6.1 or later and a x86-64 CPU with SSE4.2
support. We do not attempt to maintain a working version for Python 2.

Alchemy is intended to be run on Linux and is not officially supported on Mac
and Windows. However, it can in principle be run on any platform (though
installation may be more of a headache). In particular, on Windows, you will
need to install and run Alchemy with
[WSL](https://docs.microsoft.com/en-us/windows/wsl/about).

Note: We recommend using
[Python virtual environment](https://docs.python.org/3/tutorial/venv.html) to
mitigate conflicts with your system's Python environment.

Download and install Docker:

*   For Linux, install [Docker-CE](https://docs.docker.com/install/). Ensure
    that you can run Docker as a
    [non-root user](https://docs.docker.com/engine/install/linux-postinstall/).
*   Install Docker Desktop for
    [OSX](https://docs.docker.com/docker-for-mac/install/) or
    [Windows](https://docs.docker.com/docker-for-windows/install/).

Ensure that docker is working correctly by running `docker run -d
gcr.io/deepmind-environments/dm_alchemy:v1.0.0`.

## Installation

You can install `dm_alchemy` by cloning a local copy of our GitHub repository:

```bash
$ git clone https://github.com/deepmind/dm_alchemy.git
$ pip install wheel
$ pip install --upgrade setuptools
$ pip install ./dm_alchemy
```

To also install the dependencies for the `examples/`, install with:

```bash
$ pip install ./dm_alchemy[examples]
```

## Usage

Once `dm_alchemy` is installed, to instantiate a `dm_env` instance run the
following:

```python
import dm_alchemy

LEVEL_NAME = ('alchemy/perceptual_mapping_'
              'randomized_with_rotation_and_random_bottleneck')
settings = dm_alchemy.EnvironmentSettings(seed=123, level_name=LEVEL_NAME)
env = dm_alchemy.load_from_docker(settings)
```

For more details see the introductory colab.

[![Open in colab](https://colab.research.google.com/assets/colab-badge.svg)][Tutorial]

## Citing Alchemy

If you use Alchemy in your work, please cite the accompanying
[technical report][Paper]:

```bibtex
@article{wang2021alchemy,
    title={Alchemy: A structured task distribution for meta-reinforcement learning},
    author={Jane Wang and Michael King and Nicolas Porcel and Zeb Kurth-Nelson
        and Tina Zhu and Charlie Deck and Peter Choy and Mary Cassin and
        Malcolm Reynolds and Francis Song and Gavin Buttimore and David Reichert
        and Neil Rabinowitz and Loic Matthey and Demis Hassabis and Alex Lerchner
        and Matthew Botvinick},
    year={2021},
    journal={arXiv preprint arXiv:2102.02926},
    url={https://arxiv.org/abs/2102.02926},
}
```

## Notice

This is not an officially supported Google product.

[Tutorial]: https://colab.research.google.com/github/deepmind/dm_alchemy/blob/master/examples/AlchemyGettingStarted.ipynb
[Documentation]: docs/index.md
[Paper]: https://arxiv.org/abs/2102.02926
[Blog post]: https://deepmind.com/research/publications/Alchemy
