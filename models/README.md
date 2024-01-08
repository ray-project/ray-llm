# RayLLM model registry

Each model is defined by a YAML configuration file in this directory.

## Modify an existing model

To modify an existing model, simply edit the YAML file for that model.
Each config file consists of three sections: 

- `deployment_config`, 
- `engine_config`, 
- `scaling_config`.

It's best to check out examples of existing models to see how they are configured.

## Deployment config

The `deployment_config` section corresponds to
[Ray Serve configuration](https://docs.ray.io/en/latest/serve/production-guide/config.html)
and specifies how to [auto-scale the model](https://docs.ray.io/en/latest/serve/scaling-and-resource-allocation.html)
(via `autoscaling_config`) and what specific options you may need for your
Ray Actors during deployments (using `ray_actor_options`). We recommend using the values from our sample configuration files for `metrics_interval_s`, `look_back_period_s`, `smoothing_factor`, `downscale_delay_s` and `upscale_delay_s`. These are the configuration options you may want to modify:

* `min_replicas`, `initial_replicas`, `max_replicas` - Minimum, initial and maximum number of replicas of the model to deploy on your Ray cluster.
* `max_concurrent_queries` - Maximum number of queries that a Ray Serve replica can process at a time. Additional queries are queued at the proxy.
* `target_num_ongoing_requests_per_replica` - Guides the auto-scaling behavior. If the average number of ongoing requests across replicas is above this number, Ray Serve attempts to scale up the number of replicas, and vice-versa for downscaling. We typically set this to ~40% of the `max_concurrent_queries`.
* `ray_actor_options` - Similar to the `resources_per_worker` configuration in the `scaling_config`. Refer to the `scaling_config` section for more guidance.

### Engine config

Engine is the abstraction for interacting with a model. It is responsible for scheduling and running the model inside a Ray Actor worker group.

The `engine_config` section specifies the model ID (`model_id`), how to initialize it, and what parameters to use when generating tokens with an LLM.

RayLLM supports continuous batching, meaning incoming requests are processed as soon as they arrive, and can be added to batches that are already being processed. This means that the model is not slowed down by certain sentences taking longer to generate than others. RayLLM also supports quantization, meaning compressed models can be deployed with cheaper hardware requirements. For more details on using quantized models in RayLLM, see the [quantization guide](continuous_batching/quantization/README.md).

#### vLLM Engine Config
* `model_id` is the ID that refers to the model in the RayLLM or OpenAI API.
* `type` is the type of  inference engine. `VLLMEngine`, `TRTLLMEngine`, and `EmbeddingEngine` are currently supported.
* `engine_kwargs` and `max_total_tokens` are configuration options for the inference engine (e.g. gpu_memory_utilization, quantization, max_num_seqs and so on, see [more options](https://github.com/vllm-project/vllm/blob/main/vllm/engine/arg_utils.py#L11)). These options may vary depending on the hardware accelerator type and model size. We have tuned the parameters in the configuration files included in RayLLM for you to use as reference.
* `generation` contains configurations related to default generation parameters such as `prompt_format` and `stopping_sequences`.
* `hf_model_id` is the Hugging Face model ID. This can also be a path to a local directory. If not specified, defaults to `model_id`.
* `runtime_env` is a dictionary that contains Ray runtime environment configuration. It allows you to set per-model pip packages and environment variables. See [Ray documentation on Runtime Environments](https://docs.ray.io/en/latest/ray-core/handling-dependencies.html#runtime-environments) for more information.
* `s3_mirror_config` is a dictionary that contains configuration for loading the model from S3 instead of Hugging Face Hub. You can use this to speed up downloads.
* `gcs_mirror_config` is a dictionary that contains configuration for loading the model from Google Cloud Storage instead of Hugging Face Hub. You can use this to speed up downloads.

#### TRTLLM Engine Config
* `model_local_path` is the path to the TensorRT-LLM model directory.
* `s3_mirror_config` is a dictionary that contains configurations for loading the model from S3 instead of Hugging Face Hub. You can use this to speed up downloads.
* `generation` contains configurations related to default generation parameters such as `prompt_format` and `stopping_sequences`.
* `scheduler_policy` is to choose scheduler policy between max_utilization/guaranteed_no_evict.
(`MAX_UTILIZATION` packs as many requests as the underlying TRT engine can support in any iteration of the InflightBatching generation loop. While this is expected to maximize GPU throughput, it might require that some requests be paused and restarted depending on peak KV cache memory availability.
`GUARANTEED_NO_EVICT` uses KV cache more conservatively and guarantees that a request, once started, runs to completion without eviction.)
* `logger_level` is to configure log level for TensorRT-LLM engine. ("INFO", "ERROR", "VERBOSE", "WARNING")
* `max_num_sequences` is the maximum number of requests/sequences the backend can maintain state
* `max_tokens_in_paged_kv_cache` is to configure the maximum number of tokens in the paged kv cache.
* `kv_cache_free_gpu_mem_fraction` is to configure K-V Cache free gpu memory fraction.

#### Embedding Engine Config
* `model_id` is the ID that refers to the model in the RayLLM or OpenAI API.
* `type` is the type of inference engine. `VLLMEngine`, `TRTLLMEngine` and `EmbeddingEngine` are currently supported.
* `hf_model_id` is the Hugging Face model ID. This can also be a path to a local directory. If not specified, defaults to `model_id`.
* `max_total_tokens` is to configure number of the maximum length of each query.
* `max_batch_size` is to set the maximum batch size when queries are batched in the backend.

#### Prepare TensorRT-LLM models
You can follow the TensorRT-LLM example to generate the model.(https://github.com/NVIDIA/TensorRT-LLM/tree/v0.6.1/examples/llama). After generating the model, you can upload the model artifact to S3 and use the `s3_mirror_config` to load the model from S3. You can also place the model artifacts in a local directory and use the `model_local_path` to load the model from the local directory. See the [llama example](continuous_batching/trtllm-meta-llama--Llama-2-7b-chat-hf.yaml) for more details.


### Scaling config

Finally, the `scaling_config` section specifies what resources should be used to serve the model - this corresponds to Ray AIR [ScalingConfig](https://docs.ray.io/en/latest/train/api/doc/ray.train.ScalingConfig.html). Note that the `scaling_config` applies to each model replica, and not the entire model deployment (in other words, each replica will have `num_workers` workers).

* `num_workers` - Number of workers (i.e. Ray Actors) for each replica of the model. This controls the tensor parallelism for the model.
* `num_gpus_per_worker` - Number of GPUs to be allocated per worker. Typically, this should be 1. 
* `num_cpus_per_worker` - Number of CPUs to be allocated per worker. 
* `placement_strategy` - Ray supports different [placement strategies](https://docs.ray.io/en/latest/ray-core/scheduling/placement-group.html#placement-strategy) for guiding the physical distribution of workers. To ensure all workers are on the same node, use "STRICT_PACK".
* `resources_per_worker` - we use `resources_per_worker` to set [Ray custom resources](https://docs.ray.io/en/latest/ray-core/scheduling/resources.html#id1) and place the models on specific node types. The node resources are set in node definitions. Here are some node setup examples while using [KubeRay](https://github.com/ray-project/ray-llm/tree/master/docs/kuberay) or [Ray Clusters](https://github.com/ray-project/ray-llm/blob/master/deploy/ray/rayllm-cluster.yaml#L35). If you're deploying locally, please refer to this [guide](https://docs.ray.io/en/latest/ray-core/scheduling/resources.html#specifying-node-resources). An example configuration of `resources_per_worker` involves setting `accelerator_type_a10`: 0.01 for a Llama-2-7b model to be deployed on an A10 GPU. Note the small fraction here (0.01). The `num_gpus_per_worker` configuration along with number of GPUs available on the node will help limit the actual number of workers that Ray schedules on the node. 

If you need to learn more about a specific configuration option, or need to add a new one, don't hesitate to reach out to the team.

## Adding a new model

To add an entirely new model to the zoo, you will need to create a new YAML file.
This file should follow the naming convention 
`<organisation-name>--<model-name>-<model-parameters>-<extra-info>.yaml`. We recommend using one of the existing models as a template (ideally, one that is the same architecture as the model you are adding).

```yaml
# true by default - you can set it to false to ignore this model
# during loading
enabled: true
deployment_config:
  # This corresponds to Ray Serve settings, as generated with
  # `serve build`.
  autoscaling_config:
    min_replicas: 1
    initial_replicas: 1
    max_replicas: 8
    target_num_ongoing_requests_per_replica: 1.0
    metrics_interval_s: 10.0
    look_back_period_s: 30.0
    smoothing_factor: 1.0
    downscale_delay_s: 300.0
    upscale_delay_s: 90.0
  ray_actor_options:
    # Resources assigned to each model deployment. The deployment will be
    # initialized first, and then start prediction workers which actually hold the model.
    resources:
      accelerator_type_cpu: 0.01
engine_config:
  # Model id - this is a RayLLM id
  model_id: mosaicml/mpt-7b-instruct
  # Id of the model on Hugging Face Hub. Can also be a disk path. Defaults to model_id if not specified.
  hf_model_id: mosaicml/mpt-7b-instruct
  # LLM engine keyword arguments passed when constructing the model.
  engine_kwargs:
    trust_remote_code: true
  # Optional Ray Runtime Environment configuration. See Ray documentation for more details.
  # Add dependent libraries, environment variables, etc.
  runtime_env:
    env_vars:
      YOUR_ENV_VAR: "your_value"
  # Optional configuration for loading the model from S3 instead of Hugging Face Hub. You can use this to speed up downloads or load models not on Hugging Face Hub.
  s3_mirror_config:
    bucket_uri: s3://large-dl-models-mirror/models--mosaicml--mpt-7b-instruct/main-safetensors/
  generation:
    # Prompt format to wrap queries in. {instruction} refers to user-supplied input.
    prompt_format:
      system: "{instruction}\n"  # System message. Will default to default_system_message
      assistant: "### Response:\n{instruction}\n"  # Past assistant message. Used in chat completions API.
      trailing_assistant: "### Response:\n"  # New assistant message. After this point, model will generate tokens.
      user: "### Instruction:\n{instruction}\n"  # User message.
      default_system_message: "Below is an instruction that describes a task. Write a response that appropriately completes the request."  # Default system message.
      system_in_user: false  # Whether the system prompt is inside the user prompt. If true, the user field should include '{system}'
      add_system_tags_even_if_message_is_empty: false  # Whether to include the system tags even if the user message is empty.
      strip_whitespace: false  # Whether to automaticall strip whitespace from left and right of user supplied messages for chat completions
    # Stopping sequences. The generation will stop when it encounters any of the sequences, or the tokenizer EOS token.
    # Those can be strings, integers (token ids) or lists of integers.
    # Stopping sequences supplied by the user in a request will be appended to this.
    stopping_sequences: ["### Response:", "### End"]

# Resources assigned to each model replica. This corresponds to Ray AIR ScalingConfig.
scaling_config:
  # If using multiple GPUs set num_gpus_per_worker to be 1 and then set num_workers to be the number of GPUs you want to use.
  num_workers: 1
  num_gpus_per_worker: 1
  num_cpus_per_worker: 4
  resources_per_worker:
    # You can use custom resources to specify the instance type / accelerator type
    # to use for the model.
    accelerator_type_a10: 0.01

```

### Adding a private model

To add a private model, you can either choose to use a filesystem path or an S3/GCS mirror.

- For loading a model from file system, set `engine_config.hf_model_id` to an absolute filesystem path accessible from every node in the cluster and set `engine_config.model_id` to any ID you desire in the `organization/model` format, eg. `myorganization/llama2-finetuned`.
- For loading a model from S3 or GCS, set `engine_config.s3_mirror_config.bucket_uri` or `engine_config.gcs_mirror_config.bucket_uri` to point to a folder containing your model and tokenizer files (`config.json`, `tokenizer_config.json`, `.bin`/`.safetensors` files, etc.) and set `engine_config.model_id` to any ID you desire in the `organization/model` format, eg. `myorganization/llama2-finetuned`. The model will be downloaded to a folder in the `<TRANSFORMERS_CACHE>/models--<organization-name>--<model-name>/snapshots/<HASH>` directory on each node in the cluster. `<HASH>` will be determined by the contents of `hash` file in the S3 folder, or default to `0000000000000000000000000000000000000000`. See the [HuggingFace transformers documentation](https://huggingface.co/docs/transformers/main/en/installation#cache-setup).

For loading a model from your local file system:

```yaml
engine_config:
  model_id: YOUR_MODEL_NAME
  hf_model_id: YOUR_MODEL_LOCAL_PATH
```

For loading a model from S3:

```yaml
engine_config:
  model_id: YOUR_MODEL_NAME
  s3_mirror_config:
    bucket_uri: s3://YOUR_BUCKET_NAME/YOUR_MODEL_FOLDER

```
