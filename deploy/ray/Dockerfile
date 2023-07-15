FROM anyscale/ray:nightly-py310-cu118
SHELL ["/bin/bash", "-c"]

# Add extra dependencies
ARG DEBIAN_FRONTEND=noninteractive
RUN sudo apt-get update && sudo apt-get install -y axel nfs-common zip unzip libaio-dev git-lfs awscli curl libssl-dev gcc pkg-config libc6-dev && sudo apt-get clean

RUN pip install -i https://download.pytorch.org/whl/cu118 torch torchvision torchaudio
RUN pip uninstall -y ray && pip install -U https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-3.0.0.dev0-cp310-cp310-manylinux2014_x86_64.whl

# Created by build_aviary_wheel.sh
COPY "./dist" "/home/ray/dist"
RUN export FORCE_CUDA=1 NVCC_PREPEND_FLAGS="--forward-unknown-opts" DS_BUILD_OPS=1 DS_BUILD_AIO=0 DS_BUILD_SPARSE_ATTN=0 TORCH_CUDA_ARCH_LIST="7.0 7.2 7.5 8.0 8.6 9.0" && cd /home/ray/dist && pip install "$(ls *.whl | head -n1)[backend, frontend]"

# The build context should be the root of the repo
# So this gives the model definitions
COPY "./models" "/home/ray/models"

ENV RAY_SERVE_ENABLE_EXPERIMENTAL_STREAMING=1
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV SAFETENSORS_FAST_GPU=1

# (Optional) Verify that dependencies from the base image still work. This
# is useful for catching dependency conflicts at build time.
RUN echo "Testing Ray Import..." && python -c "import ray"
RUN ray --version
RUN jupyter --version
RUN anyscale --version
RUN sudo supervisord --version

RUN echo "Testing aviary install" && python -c "import aviary.backend"

RUN (pip cache purge || true) && conda clean -a && rm -rf ~/.cache
