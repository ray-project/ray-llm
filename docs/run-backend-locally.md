# Install and Run RayLLM Backend Locally

## Install latest Ray Serve

```bash
pip install ray[serve]
```

## Additional Dependencies

```bash
pip install redis
conda install -y -c conda-forge gxx=12.3 gxx_linux-64=12.3 libxcrypt
```

## Install RayLLM Backend from patched branch

```bash
git clone https://github.com/xwu99/ray-llm && cd ray-llm && git checkout support-vllm-cpu
pip install -e .[backend] --extra-index-url https://download.pytorch.org/whl/cpu
```

## Install latest vLLM (>= 0.4.1) on CPU

```bash
MAX_JOBS=8 VLLM_TARGET_DEVICE=cpu pip install -v git+https://github.com/vllm-project/vllm --extra-index-url https://download.pytorch.org/whl/cpu
```

## Run

Start Ray from the directory of the code:

```bash
OMP_NUM_THREADS=32 ray start --head
```

## Start Serve

Please change dtype to "bfloat16" for performance if you run on SPR machine, otherwise use "float32" for compatibility.

```bash
serve run ./serve_configs/cpu/meta-llama--Llama-2-7b-chat-hf.yaml
```

## Query

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

## List Models

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
