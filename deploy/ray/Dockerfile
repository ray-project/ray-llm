FROM rayproject/ray:nightly-cu118

RUN sudo apt-get update && sudo apt-get install -y libaio-dev git-lfs awscli && sudo rm -rf /var/lib/apt/lists/*

RUN conda install python=3.10
RUN pip install -i https://download.pytorch.org/whl/cu118 torch torchvision torchaudio

# Created by build_aviary_wheel.sh
COPY "./dist" "/home/ray/dist"
RUN cd /home/ray/dist && pip install "$(ls *.whl | head -n1)[backend]"

# The build context should be the root of the repo
# So this gives the model definitions
COPY "./models" "/home/ray/models"

ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV SAFETENSORS_FAST_GPU=1
ENV RAY_SERVE_ENABLE_EXPERIMENTAL_STREAMING=1
RUN echo "Testing aviary install" && python -c "import aviary.backend"

RUN pip cache purge && conda clean -a && rm -rf ~/.cache
