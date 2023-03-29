ARG PARENT_IMAGE
FROM $PARENT_IMAGE
ARG PYTORCH_DEPS=cpuonly
ARG PYTHON_VERSION=3.8
ARG MAMBA_DOCKERFILE_ACTIVATE=1  # (otherwise python will not be found)

# Install micromamba env and dependencies
RUN micromamba install -n base -y python=$PYTHON_VERSION \
    pytorch $PYTORCH_DEPS -c conda-forge -c pytorch -c nvidia && \
    micromamba clean --all --yes

ENV CODE_DIR /home/$MAMBA_USER

# Copy setup file only to install dependencies
COPY --chown=$MAMBA_USER:$MAMBA_USER ./setup.py ${CODE_DIR}/stable-baselines3/setup.py
COPY --chown=$MAMBA_USER:$MAMBA_USER ./stable_baselines3/version.txt ${CODE_DIR}/stable-baselines3/stable_baselines3/version.txt

RUN cd ${CODE_DIR}/stable-baselines3 && \
    pip install -e .[extra,tests,docs] && \
    # Use headless version for docker
    pip uninstall -y opencv-python && \
    pip install opencv-python-headless && \
    pip cache purge

CMD /bin/bash
