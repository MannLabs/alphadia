# syntax=docker/dockerfile:1

# https://github.com/michaelosthege/pythonnet-docker
FROM --platform=linux/amd64 mosthege/pythonnet:python3.9.16-mono6.12-pythonnet3.0.1

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/go/dockerfile-user-best-practices/
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/home/alphadiauser" \
    --shell "/sbin/nologin" \
    --uid "${UID}" \
    alphadiauser

# Download dependencies as a separate step to take advantage of Docker's caching.
# Leverage a cache mount to /root/.cache/pip to speed up subsequent builds.
# Leverage a bind mount to requirements.txt to avoid having to copy them into
# into this layer.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements/requirements.txt,target=requirements/requirements.txt \
    python -m pip install -r requirements/requirements.txt

COPY . .

RUN pip install .

USER alphadiauser

CMD alphadia --config  /app/data/config/config.yaml
