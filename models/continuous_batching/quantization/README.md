# Quantization in RayLLM

Quantization is a technique to reduce the computational and memory costs of running inference by representing the weights and/or activations with low-precision data types like 4-bit integer (int4) instead of the usual 16-bit floating point (float16).
Quantization allows users to deploy models with cheaper hardware requirements with potentially lower inference costs. 

RayLLM supports AWQ and SqueezeLLM weight-only quantization by integrating with [vLLM](https://github.com/vllm-project/vllm). Quantization can be enabled in RayLLM by specifying the `quantization` method in `engine_kwargs` and using a quantized model for `model_id` and `hf_model_id`. See the configs in this directory for quantization examples. Note that the AWQ and SqueezeLLM quantization methods in vLLM have not been fully optimized and can be slower than FP16 models for larger batch sizes. 

See the following tables for benchmarks conducted on Llama2 models using the [llmperf](https://github.com/ray-project/llmperf/) evaluation framework with vLLM 0.2.2. The quantized models were benchmarked for end-to-end (E2E) latency, time to first token (TTFT), iter-token latency (ITL), and generation throughput using default llmperf parameters.

Llama2 7B on 1 A100 80G
| Quantization Method | Mean E2E (ms) | Mean TTFT (ms) | Mean ITL (ms/token) | Mean Throughput (tok/s) |
| ------------------- | ------------- | -------------- | ------------------- | ----------------------- |
| Baseline (W16A16)   | 3212          | 362            | 18.81               | 53.44                   |
| AWQ (W4A16)         | 4148          | 994            | 21.76               | 47.09                   |
| SqueezeLLM (W4A16)  | 42372         | 13857          | 109.77              | 9.13                    |

Llama2 13B on 1 A100 80G
| Quantization Method | Mean E2E (ms) | Mean TTFT (ms) | Mean ITL (ms/token) | Mean Throughput (tok/s) |
| ------------------- | ------------- | -------------- | ------------------- | ----------------------- |
| Baseline (W16A16)   | 4371          | 644            | 31.06               | 32.25                   |
| AWQ (W4A16)         | 5626          | 1695           | 41.35               | 24.48                   |
| SqueezeLLM (W4A16)  | 64293         | 21676          | 628.71              | 5.6                     |

Llama2 70B on 4 A100 80G (SqueezeLLM Llama2 70B not available on huggingface)
| Quantization Method | Mean E2E (ms) | Mean TTFT (ms) | Mean ITL (ms/token) | Mean Throughput (tok/s) |
| ------------------- | ------------- | -------------- | ------------------- | ----------------------- |
| Baseline (W16A16)   | 8048          | 1073           | 58.1                | 17.26                   |
| AWQ (W4A16)         | 9902          | 2174           | 69.64               | 14.4                    |