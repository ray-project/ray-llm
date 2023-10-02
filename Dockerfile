# syntax=docker/dockerfile:1.4

ARG RAY_IMAGE="anyscale/ray"
ARG RAY_TAG="2.7.0oss-py39-cu118"

# Use Anyscale base image
FROM ${RAY_IMAGE}:${RAY_TAG} AS aviary

ARG RAY_HOME="/home/ray"
ARG RAY_SITE_PACKAGES_DIR="${RAY_HOME}/anaconda3/lib/python3.9/site-packages"
ARG RAY_DIST_DIR="${RAY_HOME}/dist"
ARG RAY_MODELS_DIR="${RAY_HOME}/models"
ARG RAY_UID=1000
ARG RAY_GID=100

ENV RAY_SERVE_ENABLE_NEW_HANDLE_API=1
ENV RAY_SERVE_ENABLE_EXPERIMENTAL_STREAMING=1
ENV RAY_SERVE_ENABLE_JSON_LOGGING=1

ENV FORCE_CUDA=1
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV SAFETENSORS_FAST_GPU=1

# Remove this line if we need the CUDA packages
# and NVIDIA fixes their repository #ir-gleaming-sky
RUN sudo rm -v /etc/apt/sources.list.d/cuda.list

# Install torch first
RUN pip install --no-cache-dir -U pip \
    && pip install --no-cache-dir -i https://download.pytorch.org/whl/cu118 torch torchvision torchaudio \
    && pip install --no-cache-dir tensorboard ninja

# The build context should be the root of the repo
# So this gives the model definitions
COPY --chown=${RAY_UID}:${RAY_GID} "./dist" "${RAY_DIST_DIR}"
COPY --chown=${RAY_UID}:${RAY_GID} "./models/continuous_batching" "${RAY_MODELS_DIR}/continuous_batching"
COPY --chown=${RAY_UID}:${RAY_GID} "./models/README.md" "${RAY_MODELS_DIR}/README.md"

# Install dependencies for aviary.
RUN cd "${RAY_DIST_DIR}" \
    # Update accelerate so transformers doesn't complain.
    && pip install --no-cache-dir -U accelerate \
    && pip install --no-cache-dir -U "$(ls aviary-*.whl | head -n1)[frontend,backend]" \
    # Purge caches
    && pip cache purge || true \
    && conda clean -a \
    && rm -rf ~/.cache