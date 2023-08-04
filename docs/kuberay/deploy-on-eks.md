# Deploy Aviary on Amazon EKS using KubeRay
* Note that this document will be extended to include Ray autoscaling and the deployment of multiple models in the near future.

# Part 1: Set up a Kubernetes cluster on Amazon EKS
## Step 1: Create a Kubernetes cluster on Amazon EKS

Follow the first two steps in this [AWS documentation](https://docs.aws.amazon.com/eks/latest/userguide/getting-started-console.html#)
to: (1) Create your Amazon EKS cluster (2) Configure your computer to communicate with your cluster.

## Step 2: Create node groups for the Amazon EKS cluster

You can follow "Step 3: Create nodes" in this [AWS documentation](https://docs.aws.amazon.com/eks/latest/userguide/getting-started-console.html#) to create node groups. The following section provides more detailed information.

### Create a CPU node group

Create a CPU node group for all Pods except Ray GPU workers, such as KubeRay operator, Ray head, and CoreDNS Pods.

* Create a CPU node group
  * Instance type: [**m5.xlarge**](https://aws.amazon.com/ec2/instance-types/m5/) (4 vCPU; 16 GB RAM)
  * Disk size: 256 GB
  * Desired size: 1, Min size: 0, Max size: 1

### Create a GPU node group

Create a GPU node group for Ray GPU workers.

* Create a GPU node group
  * Add a Kubernetes taint to prevent CPU Pods from being scheduled on this GPU node group
    * Key: ray.io/node-type, Value: worker, Effect: NoSchedule
  * AMI type: Bottlerocket NVIDIA (BOTTLEROCKET_x86_64_NVIDIA)
  * Instance type: [**g5.12xlarge**](https://aws.amazon.com/ec2/instance-types/g5/) (4 GPU; 96 GB GPU Memory; 48 vCPUs; 192 GB RAM)
  * Disk size: 1024 GB
  * Desired size: 1, Min size: 0, Max size: 1

Because this tutorial is for deploying 1 LLM, the maximum size of this GPU node group is 1.
If you want to deploy multiple LLMs in this cluster, you may need to increase the value of the max size.

**Warning: GPU nodes are extremely expensive. Please remember to delete the cluster if you no longer need it.**

## Step 3: Verify the node groups

If you encounter permission issues with `eksctl`, you can navigate to your AWS account's webpage and copy the
credential environment variables, including `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_SESSION_TOKEN`,
from the "Command line or programmatic access" page.

```sh
eksctl get nodegroup --cluster ${YOUR_EKS_NAME}

# CLUSTER         NODEGROUP       STATUS  CREATED                 MIN SIZE        MAX SIZE        DESIRED CAPACITY        INSTANCE TYPE   IMAGE ID                        ASG NAME                           TYPE
# ${YOUR_EKS_NAME}     cpu-node-group  ACTIVE  2023-06-05T21:31:49Z    0               1               1                       m5.xlarge       AL2_x86_64                      eks-cpu-node-group-...     managed
# ${YOUR_EKS_NAME}     gpu-node-group  ACTIVE  2023-06-05T22:01:44Z    0               1               1                       g5.12xlarge     BOTTLEROCKET_x86_64_NVIDIA      eks-gpu-node-group-...     managed
```

## Step 4: Install the DaemonSet for NVIDIA device plugin for Kubernetes

If you encounter permission issues with `kubectl`, you can follow "Step 2: Configure your computer to communicate with your cluster"
in the [AWS documentation](https://docs.aws.amazon.com/eks/latest/userguide/getting-started-console.html#).

You can refer to the [Amazon EKS optimized accelerated Amazon Linux AMIs](https://docs.aws.amazon.com/eks/latest/userguide/eks-optimized-ami.html#gpu-ami)
or [NVIDIA/k8s-device-plugin](https://github.com/NVIDIA/k8s-device-plugin) repository for more details.

```sh
# Install the DaemonSet
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.9.0/nvidia-device-plugin.yml

# Verify that your nodes have allocatable GPUs 
kubectl get nodes "-o=custom-columns=NAME:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu"

# Example output:
# NAME                                GPU
# ip-....us-west-2.compute.internal   4
# ip-....us-west-2.compute.internal   <none>
```

# Part 2: Install a KubeRay operator

```sh
# Install both CRDs and KubeRay operator v0.6.0.
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm install kuberay-operator kuberay/kuberay-operator --version 0.6.0

# It should be scheduled on the CPU node. If it is not, something is wrong.
```

At this point, you have two options:

1. You can deploy Aviary manually on a `RayCluster` (Part 3), or
2. You can deploy Aviary using a [`RayService` custom resource](https://ray-project.github.io/kuberay/guidance/rayservice/) (Part 4).

The first option is more flexible for conducting experiments.
The second option is recommended for production use due to the additional high availability features provided by the `RayService` custom resource, which will manage the underlying `RayCluster`s for you.

# Part 3: Deploy Aviary on a RayCluster (recommended for experiments)

## Step 1: Create a RayCluster with Aviary

```sh
# path: docs/kuberay
kubectl apply -f ray-cluster.aviary-eks.yaml
```

Something is worth noticing:
* The `tolerations` for workers must match the taints on the GPU node group.
    ```yaml
    # Please add the following taints to the GPU node.
    tolerations:
        - key: "ray.io/node-type"
        operator: "Equal"
        value: "worker"
        effect: "NoSchedule"
    ```
* Update `rayStartParams.resources` for Ray scheduling. The `OpenAssistant--falcon-7b-sft-top1-696.yaml` file uses both `accelerator_type_cpu` and `accelerator_type_a10`.
    ```yaml
    # Ray head: The Ray head has a Pod resource limit of 2 CPUs.
    rayStartParams:
      resources: '"{\"accelerator_type_cpu\": 2}"'

    # Ray workers: The Ray worker has a Pod resource limit of 48 CPUs and 4 GPUs.
    # `accelerator_type_a10` and `accelerator_type_a100` below are only used for Ray logical-resource scheduling.
    # This does not imply that each worker has 2 A10 GPUs and 2 A100 GPUs.
    rayStartParams:
      resources: '"{\"accelerator_type_cpu\": 48, \"accelerator_type_a10\": 2, \"accelerator_type_a100\": 2}"'
    ```

## Step 2: Deploy a LLM model with Aviary

```sh
# Step 7.1: Log in to the head Pod
export HEAD_POD=$(kubectl get pods --selector=ray.io/node-type=head -o custom-columns=POD:metadata.name --no-headers)
kubectl exec -it $HEAD_POD -- bash

# Step 7.2: Llama 2 related models require HUGGING_FACE_HUB_TOKEN to be set.
# If you don't have one, you can skip this step and deploy other models in Step 7.3.
export HUGGING_FACE_HUB_TOKEN=${YOUR_HUGGING_FACE_HUB_TOKEN}

# Step 7.3: Deploy a LLM model. You can deploy Falcon-7B if you don't have a Hugging Face Hub token.
# (1) Llama 2 7B
aviary run --model ~/models/continuous_batching/meta-llama--Llama-2-7b-chat-hf.yaml
# (2) Falcon 7B
aviary run --model ./models/continuous_batching/OpenAssistant--falcon-7b-sft-top1-696.yaml

# Step 7.3: Check the Serve application status
serve status

# [Example output]
# name: OpenAssistant--falcon-7b-sft-top1-696
# app_status:
#   status: RUNNING
#   message: ''
#   deployment_timestamp: 1691109255.5476327
# deployment_statuses:
# - name: OpenAssistant--falcon-7b-sft-top1-696_OpenAssistant--falcon-7b-sft-top1-696
#   status: HEALTHY
#   message: ''
# ---
# name: router
# app_status:
#   status: RUNNING
#   message: ''
#   deployment_timestamp: 1691109255.6641886
# deployment_statuses:
# - name: router_Router
#   status: HEALTHY
#   message: ''

# Step 7.4: List all models
export AVIARY_URL="http://localhost:8000"
aviary models

# [Example output]
# Connecting to Aviary backend at:  http://localhost:8000/
# OpenAssistant/falcon-7b-sft-top1-696

# Step 7.5: Send a query to `OpenAssistant/falcon-7b-sft-top1-696`.
aviary query --model OpenAssistant/falcon-7b-sft-top1-696 --prompt "What are the top 5 most popular programming languages?"

# [Example output for `OpenAssistant/falcon-7b-sft-top1-696`]
# Connecting to Aviary backend at:  http://localhost:8000/v1
# OpenAssistant/falcon-7b-sft-top1-696:
# The top five most popular programming languages globally, according to TIOBE, are Java, Python, C, C++, and JavaScript. However, popularity can vary by region, industry, and
# other factors. Additionally, the definition of a programming language can vary, leading to different rankings depending on the methodology used. Some rankings may include or
# exclude specific scripting languages or high-level language variants, for example.

# Here are some additional rankings of the most popular programming languages:
# * **Top 10 programming languages in 2023**: Python, JavaScript, C#, Java, PHP, TypeScript, Swift, Golang, Ruby, and Kotlin.
# [Source](https://www.toptal.com/software/programming-languages/2023-best-programming-languages/)
# * **Top 10 programming languages in 2022**: Python, JavaScript, Java, C++, C#, PHP, Swift, Kotlin, R, and TypeScript.
# [Source](https://www.toptal.com/software/programming-languages/2022-best-programming-languages/)
# * **Top 10 programming languages in 2021**: Python, JavaScript, Java, C++, C#, PHP, Swift, Go, Kotlin, and TypeScript.
# .....
# These rankings can change frequently, so it's important to keep up to date with the latest trends.
```

# Part 4: Deploy Aviary on a RayService (recommended for production)

## Step 1: Create a RayService with Aviary

```sh
# path: docs/kuberay
kubectl apply -f ray-service.aviary-eks.yaml
```

The `spec.rayClusterConfig` in `ray-service.aviary-eks.yaml` is the same as the `spec` in `ray-cluster.aviary-eks.yaml`.
The only difference lies in the `serve` port, which is required for both the Ray head and Ray worker Pods in the case of RayService.
Hence, you can refer to Part 3 for more details about how to configure the RayCluster.

> Note: Both `amazon/LightGPT` and `OpenAssistant/falcon-7b-sft-top1-696` should take about 5 minutes to become ready.
If this process takes longer, follow the instructions in [the RayService troubleshooting guide](https://github.com/ray-project/kuberay/blob/master/docs/guidance/rayservice-troubleshooting.md) to check the Ray dashboard.

```yaml
serveConfigV2: |
    applications:
    - name: amazon--LightGPT
      import_path: aviary.backend:llm_application
      route_prefix: /amazon--LightGPT
      args:
        model: "./models/continuous_batching/amazon--LightGPT.yaml"
    - name: OpenAssistant--falcon-7b-sft-top1-696
      import_path: aviary.backend:llm_application
      route_prefix: /OpenAssistant--falcon-7b-sft-top1-696
      args:
        model: "./models/continuous_batching/OpenAssistant--falcon-7b-sft-top1-696.yaml"
    - name: router
      import_path: aviary.backend:router_application
      route_prefix: /
      args:
        models:
          amazon/LightGPT: ./models/continuous_batching/amazon--LightGPT.yaml
          OpenAssistant/falcon-7b-sft-top1-696: ./models/continuous_batching/OpenAssistant--falcon-7b-sft-top1-696.yaml
```

In the YAML file, we use the `serveConfigV2` field to configure two LLM serve applications, one for LightGPT and one for Falcon-7B.
It's important to note that the `model` argument refers to the path of the LLM model's YAML file, located in the Ray head Pod.

## Step 2: Send a query to both `amazon/LightGPT` and `OpenAssistant/falcon-7b-sft-top1-696`.

Please note, there is a slight difference between deploying a model with `aviary run` and RayService.
The `router` serve application, used to support the Aviary CLI backend, will not be created by RayService. As a result, some Aviary CLI commands (e.g., `aviary query`) may cease to function.
However, this seems to be acceptable.
We can still send a query to the model via the `curl` command.

```sh
# Step 2.1: Port forward the Kubernetes serve service.
# Note that the service will be created only when all serve applications are ready.
kubectl get svc # Check if `aviary-serve-svc` is created.
kubectl port-forward service/aviary-serve-svc 8000:8000

# Step 2.2: List models via the Aviary CLI outside the Kubernetes cluster.
export AVIARY_URL="http://localhost:8000"
aviary models

# [Example output]
# Connecting to Aviary backend at:  http://localhost:8000/v1
# OpenAssistant/falcon-7b-sft-top1-696
# amazon/LightGPT

# Step 2.3: Send a query to `amazon/LightGPT`.
aviary query --model amazon/LightGPT --prompt "What are the top 5 most popular programming languages?"

# [Example output]
# Connecting to Aviary backend at:  http://localhost:8000/v1
# amazon/LightGPT:
# 1. Java
# 2. C++
# 3. JavaScript
# 4. Python
# 5. SQL

# Step 2.4: Send a query to `OpenAssistant/falcon-7b-sft-top1-696`.
aviary query --model OpenAssistant/falcon-7b-sft-top1-696 --prompt "What are the top 5 most popular programming languages?"

# [Example output for `OpenAssistant/falcon-7b-sft-top1-696`]
# Connecting to Aviary backend at:  http://localhost:8000/v1
# OpenAssistant/falcon-7b-sft-top1-696:
# The top five most popular programming languages globally, according to TIOBE, are Java, Python, C, C++, and JavaScript. However, popularity can vary by region, industry, and
# other factors. Additionally, the definition of a programming language can vary, leading to different rankings depending on the methodology used. Some rankings may include or
# exclude specific scripting languages or high-level language variants, for example.

# Here are some additional rankings of the most popular programming languages:
# * **Top 10 programming languages in 2023**: Python, JavaScript, C#, Java, PHP, TypeScript, Swift, Golang, Ruby, and Kotlin.
# [Source](https://www.toptal.com/software/programming-languages/2023-best-programming-languages/)
# * **Top 10 programming languages in 2022**: Python, JavaScript, Java, C++, C#, PHP, Swift, Kotlin, R, and TypeScript.
# [Source](https://www.toptal.com/software/programming-languages/2022-best-programming-languages/)
# * **Top 10 programming languages in 2021**: Python, JavaScript, Java, C++, C#, PHP, Swift, Go, Kotlin, and TypeScript.
# .....
# These rankings can change frequently, so it's important to keep up to date with the latest trends.

# Step 2.5: Send a query to `OpenAssistant/falcon-7b-sft-top1-696` and get streaming response.
aviary stream --model OpenAssistant/falcon-7b-sft-top1-696 --prompt "What are the top 5 most popular programming languages?"
```

# Part 5: Clean up resources

**Warning: GPU nodes are extremely expensive. Please remember to delete the cluster if you no longer need it.**

```sh
# path: docs/kuberay
# Case 1: Aviary was deployed on a RayCluster
kubectl delete -f ray-cluster.aviary-eks.yaml
# Case 2: Aviary was deployed as a RayService
kubectl delete -f ray-service.aviary-eks.yaml

# Uninstall the KubeRay operator chart
helm uninstall kuberay-operator

# Delete the Amazon EKS cluster via AWS Web UI
```
