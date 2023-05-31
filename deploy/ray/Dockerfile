FROM rayproject/ray-ml:nightly-gpu

ENV HF_HUB_ENABLE_HF_TRANSFER=1

RUN sudo apt-get update
RUN sudo apt-get install -y libaio-dev git-lfs awscli

RUN pip install --upgrade pip
RUN pip uninstall -y ray torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric tensorflow
RUN conda install python=3.10
RUN pip install "ray[default,serve] @ https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-3.0.0.dev0-cp310-cp310-manylinux2014_x86_64.whl"
COPY "./dist" "/home/ray/dist"
RUN cd /home/ray/dist && pip install "$(ls *.whl | head -n1)[backend]"


COPY "./deploy/ray/backend.yaml" "/home/ray/abcd"

# The build context should be the root of the repo
# So this gives the model definitions
COPY "./models" "/home/ray/models"

RUN echo "Testing aviary install" && python -c "import aviary.backend"