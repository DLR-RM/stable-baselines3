ARG PARENT_IMAGE
FROM $PARENT_IMAGE
ARG PYTORCH_DEPS=cpuonly
ARG PYTHON_VERSION=3.10
# (otherwise python will not be found)
ARG MAMBA_DOCKERFILE_ACTIVATE=1
USER root
# Install micromamba env and dependencies
RUN micromamba install -n base -y python=$PYTHON_VERSION \
    pytorch $PYTORCH_DEPS -c conda-forge -c pytorch -c nvidia && \
    micromamba clean --all --yes

RUN apt-get update
RUN apt-get install -y build-essential

ENV CODE_DIR /home/$MAMBA_USER
WORKDIR ${CODE_DIR}/stable-baselines3
# Copy setup file only to install dependencies
COPY --chown=$MAMBA_USER:$MAMBA_USER ./poetry.lock .
COPY --chown=$MAMBA_USER:$MAMBA_USER ./pyproject.toml .

RUN pip install poetry>=1.6.1
RUN poetry install --all-extras --no-root --no-cache
RUN pip uninstall -y opencv-python
RUN pip install opencv-python-headless
RUN pip cache purge

CMD /bin/bash
