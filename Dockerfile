ARG PARENT_IMAGE
FROM $PARENT_IMAGE
ARG PYTORCH_DEPS=cpuonly
ARG PYTHON_VERSION=3.6

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         ca-certificates \
         libjpeg-dev \
         libpng-dev && \
     rm -rf /var/lib/apt/lists/*

# Install anaconda abd dependencies
RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install -y python=$PYTHON_VERSION numpy pyyaml scipy ipython mkl mkl-include && \
     /opt/conda/bin/conda install -y pytorch $PYTORCH_DEPS -c pytorch && \
     /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH

ENV CODE_DIR /root/code

# Copy setup file only to install dependencies
COPY ./setup.py ${CODE_DIR}/stable-baselines3/setup.py
COPY ./stable_baselines3/version.txt ${CODE_DIR}/stable-baselines3/stable_baselines3/version.txt

RUN \
    cd ${CODE_DIR}/stable-baselines3 3&& \
    pip install -e .[extra,tests,docs] && \
    # Use headless version for docker
    pip install opencv-python-headless && \
    rm -rf $HOME/.cache/pip


# Codacy deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    default-jre \
    jq && \
    rm -rf /var/lib/apt/lists/*

# Codacy code coverage report: used for partial code coverage reporting
RUN cd $CODE_DIR && \
    curl -Ls -o codacy-coverage-reporter.jar "$(curl -Ls https://api.github.com/repos/codacy/codacy-coverage-reporter/releases/latest | jq -r '.assets | map({name, browser_download_url} | select(.name | (startswith("codacy-coverage-reporter") and contains("assembly") and endswith(".jar")))) | .[0].browser_download_url')"

CMD /bin/bash
