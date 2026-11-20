# Opt-in CUDA variant of vsim-ci.Dockerfile, for hosts with a GPU and the NVIDIA container toolkit.
# CI does not build it by default; .github/workflows/build_docker_cuda.yml builds it on manual
# dispatch.
FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all
WORKDIR /app

# xvfb, xauth and mesa's software driver provide the headless display Blender's viewport and
# playblast renderer need; llvmpipe is the fallback when no GPU is passed through.
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential curl ca-certificates automake autoconf pkg-config \
    libxi6 libxkbcommon-x11-0 libglfw3-dev libgles2-mesa-dev libsm6 \
    xvfb xauth libgl1-mesa-dri \
    ffmpeg git nano bzip2 xz-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY blender_dist /usr/local/blender
ENV PATH="$PATH:/usr/local/blender/"

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV PATH="/root/.local/bin:${PATH}"

# Copy the interpreter pin first: a bare `uv venv` would otherwise grab the newest Python it can
# find, and the source tree (and thus .python-version) only lands at the later `COPY . /src`.
COPY .python-version /app/
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG TORCH_BACKEND=cu128
RUN uv pip install --no-cache torch torchvision --index-url "$TORCH_INDEX_URL"

COPY . /src
RUN uv pip install --no-cache --torch-backend "$TORCH_BACKEND" --editable /src
# Blender ships its own interpreter, so `post-install` needs the CUDA index again for its torch.
RUN visionsim post-install --editable --torch-index-url "$TORCH_INDEX_URL"

# Blender's viewport renderer needs an X display with a GL context; the entrypoint starts a
# virtual one before running the container's command.
COPY .github/workflows/xvfb-entrypoint.sh /usr/local/bin/xvfb-entrypoint.sh
RUN chmod +x /usr/local/bin/xvfb-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/xvfb-entrypoint.sh"]
CMD ["bash"]
