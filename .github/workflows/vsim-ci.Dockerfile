# Use a slim Debian image as our base
FROM debian:trixie-slim
WORKDIR /app 

# Install system dependencies needed for both Visionsim and Blender.
# `xvfb` (plus `xauth` and mesa's software `libgl1-mesa-dri` driver) provides the headless virtual
# display with a (llvmpipe) GL context that Blender's viewport/playblast renderer needs.
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential curl ca-certificates automake autoconf pkg-config \
    libxi6 libxkbcommon-x11-0 libglfw3-dev libgles2-mesa-dev libsm6 \
    xvfb xauth libgl1-mesa-dri \
    ffmpeg git nano bzip2 xz-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the pre-downloaded blender distribution provided by GitHub Actions
# It is expected to be placed locally in `blender_dist` by the CI workflow.
COPY blender_dist /usr/local/blender
ENV PATH="$PATH:/usr/local/blender/"

# Install uv, the fast Python package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV PATH="/root/.local/bin:${PATH}"

# Create venv and "activate" it by placing it in PATH
# Copy the interpreter pin first: a bare `uv venv` would otherwise grab the newest Python it can
# find, and the source tree (and thus .python-version) only lands at the later `COPY . /src`.
COPY .python-version /app/
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install python dependencies
# The image has no GPU, so torch must come from PyTorch's CPU index; the default PyPI Linux
# wheel pulls in multi-GB CUDA runtime packages.
# Using --no-cache prevents uv from caching wheels in the container
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN uv pip install --no-cache torch torchvision --index-url "$TORCH_INDEX_URL"

# Install visionsim package as editable, ensures the current commit is pulled and installed.
COPY . /src
RUN uv pip install --no-cache --torch-backend cpu --editable /src
# Blender ships its own interpreter, so `post-install` needs the CPU index again for its torch.
RUN visionsim post-install --editable --torch-index-url "$TORCH_INDEX_URL"

# Blender's viewport renderer (playblast) needs an X display with a GL context, which containers
# lack. The entrypoint starts a virtual one with Xvfb before running the container's command.
COPY .github/workflows/xvfb-entrypoint.sh /usr/local/bin/xvfb-entrypoint.sh
RUN chmod +x /usr/local/bin/xvfb-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/xvfb-entrypoint.sh"]
CMD ["bash"]
