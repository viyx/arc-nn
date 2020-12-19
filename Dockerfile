# For more information, please refer to https://aka.ms/vscode-docker-python
# FROM python:3.8-slim-buster
FROM gcr.io/tpu-pytorch/xla:nightly_3.6

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1
# RUN update-alternatives --install /usr/lib/python3.5 python /usr/lib/python3 1
# Install pip requirements
ADD requirements.txt .
RUN python -m pip install -r requirements.txt

WORKDIR /app
ADD . /app

# Switching to a non-root user, please refer to https://aka.ms/vscode-docker-python-user-rights
# RUN useradd appuser && chown -R appuser /app
# USER appuser

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["python", "train.py"]
