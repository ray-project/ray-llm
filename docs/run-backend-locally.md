# Install and Run RayLLM Backend Locally

## Install latest Ray Serve

```bash
pip install ray[serve]
```

## Install RayLLM Backend from PR #149 branch (TODO: update link when PR merged)

```bash
git clone https://github.com/xwu99/ray-llm && cd ray-llm && git checkout support-vllm-cpu
```

Install for GPU device:
```bash
pip install -e .[backend]
```

Install for CPU device:
```bash
pip install -e .[backend] --extra-index-url https://download.pytorch.org/whl/cpu
```

## (Optional) Additional steps to install vllm from source for CPU device

### Install GCC (>=12.3)

```bash
conda install -y -c conda-forge gxx=12.3 gxx_linux-64=12.3 libxcrypt
```

### Install latest vLLM (>= 0.4.1) on CPU

```bash
MAX_JOBS=8 VLLM_TARGET_DEVICE=cpu pip install -v git+https://github.com/vllm-project/vllm --extra-index-url https://download.pytorch.org/whl/cpu
```

## Test Run

### Run on CPU device

Start Ray from the directory of the code:

```bash
OMP_NUM_THREADS=32 ray start --head
```

To start serving:

__Notice: Please change dtype to "bfloat16" for performance if you run on 4th generation Xeon Scalable (codename "Sapphire Rapids") or later CPU, otherwise use "float32" for compatibility.__

```bash
serve run ./serve_configs/cpu/meta-llama--Llama-2-7b-chat-hf.yaml
```

### Run on GPU device

Start Ray from the directory of the code:

```bash
ray start --head
```

To start serving:

__Notice: Please change "accelerator_type_a10" to match your GPU type__

```bash
serve run ./serve_configs/meta-llama--Llama-2-7b-chat-hf.yaml
```

### Query

Export the endpoint URL:

```bash
export ENDPOINT_URL="http://localhost:8000/v1"
```

Send a POST request for chat completions:

```bash
curl -X POST $ENDPOINT_URL/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-2-7b-chat-hf",
    "messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello!"}],
    "temperature": 0.7
  }'
```

### List Models

Set the API base and key:

```bash
export OPENAI_API_BASE="http://localhost:8000/v1"
export OPENAI_API_KEY="not-a-key"
```

List available RayLLM models:

```bash
rayllm models
```

## Caveat

- The current working directory is where `ray start --head` runs, therefore if you use relative paths to define `models/*` files, the path should be relative to where Ray starts.
- When switching Conda environments, you need to restart Ray.
