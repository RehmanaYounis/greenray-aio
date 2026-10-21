# GreenRay AIO: run with
#   docker run -it --rm -v greenray:/home/lab/.greenray ghcr.io/rehmanayounis/greenray-aio
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS build
WORKDIR /src
COPY pyproject.toml uv.lock README.md LICENSE NOTICE ./
COPY src ./src
RUN uv sync --locked --no-dev --no-editable

FROM python:3.12-slim-bookworm
# nano and vim back the simulator's editors; less is for man pages
RUN apt-get update \
    && apt-get install -y --no-install-recommends nano vim-tiny less \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash lab
COPY --from=build --chown=lab:lab /src/.venv /opt/greenray
USER lab
WORKDIR /home/lab
ENV PATH="/opt/greenray/bin:$PATH" \
    TERM=xterm-256color \
    EDITOR=nano
VOLUME ["/home/lab/.greenray"]
ENTRYPOINT ["greenray"]
