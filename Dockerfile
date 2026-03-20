ARG PARENT_IMAGE=mambaorg/micromamba:2.0-ubuntu24.04
FROM $PARENT_IMAGE
ARG PYTORCH_DEPS=https://download.pytorch.org/whl/cpu
ARG PYTHON_VERSION=3.12
ARG MAMBA_DOCKERFILE_ACTIVATE=1  # (otherwise python will not be found)

# Install micromamba env and dependencies
RUN micromamba install -n base -y python=$PYTHON_VERSION && \
    micromamba clean --all --yes

ENV CODE_DIR=/home/$MAMBA_USER

# Copy setup file only to install dependencies
COPY --chown=$MAMBA_USER:$MAMBA_USER ./setup.py ${CODE_DIR}/stable-baselines3/setup.py
COPY --chown=$MAMBA_USER:$MAMBA_USER ./stable_baselines3/version.txt ${CODE_DIR}/stable-baselines3/stable_baselines3/version.txt

RUN cd ${CODE_DIR}/stable-baselines3 && \
    pip install uv && \
    uv pip install --system torch --default-index ${PYTORCH_DEPS} && \
    uv pip install --system -e .[extra,tests,docs] && \
    # Use headless version for docker
    uv pip uninstall opencv-python && \
    uv pip install --system opencv-python-headless && \
    pip cache purge && \
    uv cache clean

CMD /bin/bash
