# For more information, please refer to https://aka.ms/vscode-docker-python
# FROM python:3.8-slim-buster
# FROM gcr.io/tpu-pytorch/xla:nightly_3.7
FROM gcr.io/tpu-pytorch/xla:nightly_3.8_tpuvm

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1
# RUN update-alternatives --install /usr/lib/python3.5 python /usr/lib/python3 1
# Install pip requirements
# ADD requirements.txt .
# RUN python -m pip install -r requirements.txt

WORKDIR /app
ADD . /app

ARG USERNAME=vitaliy
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    # && apt-get updat/e \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************

# [Optional] Set the default user. Omit if you want to keep the default as root.
# USER root


# Switching to a non-root user, please refer to https://aka.ms/vscode-docker-python-user-rights
# RUN useradd appuser && chown -R appuser /app
# USER appuser

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["python", "train.py"]
