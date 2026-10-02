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
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install python dependencies
# Here we install pytorch manually to ensure it's cpu-only
# Using --no-cache prevents uv from caching wheels in the container
RUN uv pip install --no-cache torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install visionsim package as editable, ensures the current commit is pulled and installed.
COPY . /src
RUN uv pip install --no-cache --editable /src
RUN visionsim post-install --editable

# Blender's viewport renderer (playblast) needs an X display with a GL context, which containers
# lack. The entrypoint starts a virtual one with Xvfb before running the container's command.
COPY .github/workflows/xvfb-entrypoint.sh /usr/local/bin/xvfb-entrypoint.sh
RUN chmod +x /usr/local/bin/xvfb-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/xvfb-entrypoint.sh"]
CMD ["bash"]
