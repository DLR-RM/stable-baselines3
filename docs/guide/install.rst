.. _install:

Installation
============


Prerequisites
-------------

Stable-Baselines3 requires python 3.8+ and PyTorch >= 1.13

Windows 10
~~~~~~~~~~

We recommend using `Anaconda <https://conda.io/docs/user-guide/install/windows.html>`_ for Windows users for easier installation of Python packages and required libraries. You need an environment with Python version 3.6 or above.

For a quick start you can move straight to installing Stable-Baselines3 in the next step.

.. note::

	Trying to create Atari environments may result to vague errors related to missing DLL files and modules. This is an
	issue with atari-py package. `See this discussion for more information <https://github.com/openai/atari-py/issues/65>`_.


Stable Release
~~~~~~~~~~~~~~
To install Stable Baselines3 with pip, execute:

.. code-block:: bash

    pip install stable-baselines3[extra]

.. note::
        Some shells such as Zsh require quotation marks around brackets, i.e. ``pip install 'stable-baselines3[extra]'`` `More information <https://stackoverflow.com/a/30539963>`_.


This includes an optional dependencies like Tensorboard, OpenCV or ``ale-py`` to train on atari games. If you do not need those, you can use:

.. code-block:: bash

    pip install stable-baselines3


.. note::

  If you need to work with OpenCV on a machine without a X-server (for instance inside a docker image),
  you will need to install ``opencv-python-headless``, see `issue #298 <https://github.com/DLR-RM/stable-baselines3/issues/298>`_.


Bleeding-edge version
---------------------

.. code-block:: bash

	pip install git+https://github.com/DLR-RM/stable-baselines3

with extras:

.. code-block:: bash

  pip install "stable_baselines3[extra,tests,docs] @ git+https://github.com/DLR-RM/stable-baselines3"


Development version
-------------------

To contribute to Stable-Baselines3, with support for running tests and building the documentation.

.. code-block:: bash

    git clone https://github.com/DLR-RM/stable-baselines3 && cd stable-baselines3
    pip install -e .[docs,tests,extra]


Using Docker Images
-------------------

If you are looking for docker images with stable-baselines already installed in it,
we recommend using images from `RL Baselines3 Zoo <https://github.com/DLR-RM/rl-baselines3-zoo>`_.

Otherwise, the following images contained all the dependencies for stable-baselines3 but not the stable-baselines3 package itself.
They are made for development.

Use Built Images
~~~~~~~~~~~~~~~~

GPU image (requires `nvidia-docker`_):

.. code-block:: bash

   docker pull stablebaselines/stable-baselines3

CPU only:

.. code-block:: bash

   docker pull stablebaselines/stable-baselines3-cpu

Build the Docker Images
~~~~~~~~~~~~~~~~~~~~~~~~

Build GPU image (with nvidia-docker):

.. code-block:: bash

   make docker-gpu

Build CPU image:

.. code-block:: bash

   make docker-cpu

Note: if you are using a proxy, you need to pass extra params during
build and do some `tweaks`_:

.. code-block:: bash

   --network=host --build-arg HTTP_PROXY=http://your.proxy.fr:8080/ --build-arg http_proxy=http://your.proxy.fr:8080/ --build-arg HTTPS_PROXY=https://your.proxy.fr:8080/ --build-arg https_proxy=https://your.proxy.fr:8080/

Run the images (CPU/GPU)
~~~~~~~~~~~~~~~~~~~~~~~~

Run the nvidia-docker GPU image

.. code-block:: bash

   docker run -it --runtime=nvidia --rm --network host --ipc=host --name test --mount src="$(pwd)",target=/home/mamba/stable-baselines3,type=bind stablebaselines/stable-baselines3 bash -c 'cd /home/mamba/stable-baselines3/ && pytest tests/'

Or, with the shell file:

.. code-block:: bash

   ./scripts/run_docker_gpu.sh pytest tests/

Run the docker CPU image

.. code-block:: bash

   docker run -it --rm --network host --ipc=host --name test --mount src="$(pwd)",target=/home/mamba/stable-baselines3,type=bind stablebaselines/stable-baselines3-cpu bash -c 'cd /home/mamba/stable-baselines3/ && pytest tests/'

Or, with the shell file:

.. code-block:: bash

   ./scripts/run_docker_cpu.sh pytest tests/

Explanation of the docker command:

-  ``docker run -it`` create an instance of an image (=container), and
   run it interactively (so ctrl+c will work)
-  ``--rm`` option means to remove the container once it exits/stops
   (otherwise, you will have to use ``docker rm``)
-  ``--network host`` don't use network isolation, this allow to use
   tensorboard/visdom on host machine
-  ``--ipc=host`` Use the host system’s IPC namespace. IPC (POSIX/SysV IPC) namespace provides
   separation of named shared memory segments, semaphores and message
   queues.
-  ``--name test`` give explicitly the name ``test`` to the container,
   otherwise it will be assigned a random name
-  ``--mount src=...`` give access of the local directory (``pwd``
   command) to the container (it will be map to ``/home/mamba/stable-baselines``), so
   all the logs created in the container in this folder will be kept
-  ``bash -c '...'`` Run command inside the docker image, here run the tests
   (``pytest tests/``)

.. _nvidia-docker: https://github.com/NVIDIA/nvidia-docker
.. _tweaks: https://stackoverflow.com/questions/23111631/cannot-download-docker-images-behind-a-proxy
