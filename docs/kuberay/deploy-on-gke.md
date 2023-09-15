# Deploy RayLLM on Google Kubernetes Engine (GKE) using KubeRay

In this tutorial, we will:

1. Set up a Kubernetes cluster on GKE.
2. Deploy the KubeRay operator and a Ray cluster on GKE.
3. Run an LLM model with Ray Serve.

* Note that this document will be extended to include Ray autoscaling and the deployment of multiple models in the near future.

## Step 1: Create a Kubernetes cluster on GKE

Run this command and all following commands on your local machine or on the [Google Cloud Shell](https://cloud.google.com/shell). If running from your local machine, you will need to install the [Google Cloud SDK](https://cloud.google.com/sdk/docs/install).

```sh
gcloud container clusters create rayllm-gpu-cluster \
    --num-nodes=1 --min-nodes 0 --max-nodes 1 --enable-autoscaling \
    --zone=us-west1-b --machine-type e2-standard-8
```

This command creates a Kubernetes cluster named `rayllm-gpu-cluster` with 1 node in the `us-west1-b` zone. In this example, we use the `e2-standard-8` machine type, which has 8 vCPUs and 32 GB RAM. The cluster has autoscaling enabled, so the number of nodes can increase or decrease based on the workload.

You can also create a cluster from the [Google Cloud Console](https://console.cloud.google.com/kubernetes/list).

## Step 2: Create a GPU node pool

Run the following command to create a GPU node pool for Ray GPU workers.
(You can also create it from the Google Cloud Console; see the [GKE documentation](https://cloud.google.com/kubernetes-engine/docs/how-to/node-taints#create_a_node_pool_with_node_taints) for more details.)

```sh
gcloud container node-pools create gpu-node-pool \
  --accelerator type=nvidia-l4-vws,count=4 \
  --zone us-west1-b \
  --cluster rayllm-gpu-cluster \
  --num-nodes 1 \
  --min-nodes 0 \
  --max-nodes 1 \
  --enable-autoscaling \
  --machine-type g2-standard-48 \
  --node-taints=ray.io/node-type=worker:NoSchedule 
```

The `--accelerator` flag specifies the type and number of GPUs for each node in the node pool. In this example, we use the [NVIDIA L4](https://cloud.google.com/compute/docs/gpus#l4-gpus) GPU. The machine type `g2-standard-48` has 4 GPUs, 48 vCPUs and 192 GB RAM.

Because this tutorial is for deploying 1 LLM, the maximum size of this GPU node pool is 1.
If you want to deploy multiple LLMs in this cluster, you may need to increase the value of the max size.

The taint `ray.io/node-type=worker:NoSchedule` prevents CPU-only Pods such as the Kuberay operator, Ray head, and CoreDNS pods from being scheduled on this GPU node pool. This is because GPUs are expensive, so we want to use this node pool for Ray GPU workers only.

Concretely, any Pod that does not have the following toleration will not be scheduled on this GPU node pool:

```yaml
tolerations:
- key: ray.io/node-type
  operator: Equal
  value: worker
  effect: NoSchedule
```

This toleration has already been added to the RayCluster YAML manifest `ray-cluster.aviary-gke.yaml` used in Step 6.

For more on taints and tolerations, see the [Kubernetes documentation](https://kubernetes.io/docs/concepts/scheduling-eviction/taint-and-toleration/).

## Step 3: Configure `kubectl` to connect to the cluster

Run the following command to download credentials and configure the Kubernetes CLI to use them.

```sh
gcloud container clusters get-credentials rayllm-gpu-cluster --zone us-west1-b
```

For more details, see the [GKE documentation](https://cloud.google.com/kubernetes-engine/docs/how-to/cluster-access-for-kubectl).

## Step 4: Install NVIDIA GPU device drivers

This step is required for GPU support on GKE. See the [GKE documentation](https://cloud.google.com/kubernetes-engine/docs/how-to/gpus#installing_drivers) for more details.

```sh
# Install NVIDIA GPU device driver
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded-latest.yaml

# Verify that your nodes have allocatable GPUs 
kubectl get nodes "-o=custom-columns=NAME:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu"

# Example output:
# NAME                                GPU
# ...                                 4
# ...                                 <none>
```

## Step 5: Install the KubeRay operator

```sh
# Install both CRDs and KubeRay operator v0.5.0.
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm install kuberay-operator kuberay/kuberay-operator --version 0.5.0

# It should be scheduled on the CPU node. If it is not, something is wrong.
```

## Step 6: Create a RayCluster with RayLLM

If you are running this tutorial on the Google Cloud Shell, please copy the file `docs/kuberay/ray-cluster.aviary-gke.yaml` to the Google Cloud Shell. You may find it useful to use the [Cloud Shell Editor](https://cloud.google.com/shell/docs/editor-overview) to edit the file.

Now you can create a RayCluster with RayLLM. RayLLM is included in the image `anyscale/aviary:latest`, which is specified in the RayCluster YAML manifest `ray-cluster.aviary-gke.yaml`.

```sh
# path: docs/kuberay
kubectl apply -f ray-cluster.aviary-gke.yaml
```

Note the following aspects of the YAML file:

* The `tolerations` for workers match the taints we specified in Step 2. This ensures that the Ray GPU workers are scheduled on the GPU node pool.

    ```yaml
    # Please add the following taints to the GPU node.
    tolerations:
        - key: "ray.io/node-type"
        operator: "Equal"
        value: "worker"
        effect: "NoSchedule"
    ```

* The field `rayStartParams.resources` has been configured to allow Ray to schedule Ray tasks and actors appropriately. The `mosaicml--mpt-7b-chat.yaml` file uses two [custom resources](https://docs.ray.io/en/latest/ray-core/scheduling/resources.html#custom-resources), `accelerator_type_cpu` and `accelerator_type_a10`.  See [the Ray documentation](https://docs.ray.io/en/latest/ray-core/scheduling/resources.html) for more details on resources.

    ```yaml
    # Ray head: The Ray head has a Pod resource limit of 2 CPUs.
    rayStartParams:
      resources: '"{\"accelerator_type_cpu\": 2}"'

    # Ray workers: The Ray worker has a Pod resource limit of 48 CPUs and 4 GPUs.
    rayStartParams:
        resources: '"{\"accelerator_type_cpu\": 48, \"accelerator_type_a10\": 4}"'
    ```

## Step 7: Deploy an LLM model with RayLLM

```sh
# Step 7.1: Log in to the head Pod
export HEAD_POD=$(kubectl get pods --selector=ray.io/node-type=head -o custom-columns=POD:metadata.name --no-headers)
kubectl exec -it $HEAD_POD -- bash

# Step 7.2: Deploy the `meta-llama/Llama-2-7b-chat-hf` model
serve run serve/meta-llama--Llama-2-7b-chat-hf.yaml

# Step 7.3: Check the Serve application status
serve status

# [Example output]
# proxies:
#   e4dc8d29f19e3900c0b93dabb76ce9bcc6f42e36bdf5484ca57ec774: HEALTHY
#   4f4edf80bf644846175eec0a4daabb3f3775e64738720b6b2ae5c139: HEALTHY
# applications:
#   router:
#     status: RUNNING
#     message: ''
#     last_deployed_time_s: 1694808658.0861287
#     deployments:
#       Router:
#         status: HEALTHY
#         replica_states:
#           RUNNING: 2
#         message: ''
#   meta-llama--Llama-2-7b-chat-hf:
#     status: RUNNING
#     message: ''
#     last_deployed_time_s: 1694808658.0861287
#     deployments:
#       meta-llama--Llama-2-7b-chat-hf:
#         status: HEALTHY
#         replica_states:
#           RUNNING: 1
#         message: ''

# Step 7.4: Check the live Serve app's config
serve config

# [Example output]
# name: router
# route_prefix: /
# import_path: aviary.backend:router_application
# args:
#   models:
#     meta-llama/Llama-2-7b-chat-hf: ./models/continuous_batching/meta-llama--Llama-2-7b-chat-hf.yaml

# ---

# name: meta-llama--Llama-2-7b-chat-hf
# route_prefix: /meta-llama--Llama-2-7b-chat-hf
# import_path: aviary.backend:llm_application
# args:
#   model: ./models/continuous_batching/meta-llama--Llama-2-7b-chat-hf.yaml

# Step 7.5: Send a query to `meta-llama/Llama-2-7b-chat-hf`.
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-2-7b-chat-hf",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What are the top 5 most popular programming languages?"}
    ],
    "temperature": 0.7
  }'

# [Example output for `meta-llama/Llama-2-7b-chat-hf`]
{
  "id":"meta-llama/Llama-2-7b-chat-hf-95239f0b-4601-4557-8a33-3977e9b6b779",
  "object":"text_completion","created":1694814804,"model":"meta-llama/Llama-2-7b-chat-hf",
  "choices":[
    {
      "message":
      {
        "role":"assistant",
        "content":"As a helpful assistant, I'm glad to provide you with the top 5 most popular programming languages based on various sources and metrics:\n\n1. Java: Java is a popular language used for developing enterprise-level applications, Android apps, and web applications. It's known for its platform independence, which allows Java developers to create applications that can run on any device supporting the Java Virtual Machine (JVM).\n\n2. Python: Python is a versatile language that's widely used in various industries, including web development, data science, artificial intelligence, and machine learning. Its simplicity, readability, and ease of use make it a favorite among developers.\n\n3. JavaScript: JavaScript is the language of the web and is used for creating interactive client-side functionality for web applications. It's also used in mobile app development, game development, and server-side programming.\n\n4. C++: C++ is a high-performance language used for developing operating systems, games, and other high-performance applications. It's known for its efficiency, speed, and flexibility, making it a popular choice among developers.\n\n5. PHP: PHP is a server-side scripting language used for web development, especially for building dynamic websites and web applications. It's known for its ease of use and is widely used in the web development community.\n\nThese are the top 5 most popular programming languages based on various sources, but it's worth noting that programming language popularity can vary depending on the source and the time frame considered."
      },
      "index":0,
      "finish_reason":"stop"
    }
  ],
  "usage":{
    "prompt_tokens":39,
    "completion_tokens":330,
    "total_tokens":369
  }
}
```

## Step 8: Clean up resources

**Warning: GPU nodes are extremely expensive. Please remember to delete the cluster if you no longer need it.**

```sh
# Step 8.1: Delete the RayCluster
# path: docs/kuberay
kubectl delete -f ray-cluster.aviary-gke.yaml

# Step 8.2: Uninstall the KubeRay operator chart
helm uninstall kuberay-operator

# Step 8.3: Delete the GKE cluster
gcloud container clusters delete rayllm-gpu-cluster
```

See the [GKE documentation](https://cloud.google.com/kubernetes-engine/docs/how-to/deleting-a-cluster) for more details on deleting a GKE cluster.
