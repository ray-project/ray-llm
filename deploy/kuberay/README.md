# Deploy Aviary on Amazon EKS using KubeRay
* Note that this document will be extended to include Ray autoscaling and the deployment of multiple models in the near future.

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

## Step 5: Install a KubeRay operator

```sh
# Install both CRDs and KubeRay operator v0.5.0.
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm install kuberay-operator kuberay/kuberay-operator --version 0.5.0

# It should be scheduled on the CPU node. If it is not, something is wrong.
```

## Step 6: Create a RayCluster with Aviary

```sh
# path: deploy/kuberay
kubectl apply -f kuberay.yaml
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
* Update `rayStartParams.resources` for Ray scheduling. The `mosaicml--mpt-7b-chat.yaml` file uses both `accelerator_type_cpu` and `accelerator_type_a10`.
    ```yaml
    # Ray head: The Ray head has a Pod resource limit of 2 CPUs.
    rayStartParams:
      resources: '"{\"accelerator_type_cpu\": 2}"'

    # Ray workers: The Ray worker has a Pod resource limit of 48 CPUs and 4 GPUs.
    rayStartParams:
        resources: '"{\"accelerator_type_cpu\": 48, \"accelerator_type_a10\": 4}"'
    ```

## Step 7: Deploy a LLM model with Aviary

```sh
# Step 7.1: Log in to the head Pod
export HEAD_POD=$(kubectl get pods --selector=ray.io/node-type=head -o custom-columns=POD:metadata.name --no-headers)
kubectl exec -it $HEAD_POD -- bash

# Step 7.2: Deploy a `mosaicml/mpt-7b-chat` model
aviary run --model ./models/mosaicml--mpt-7b-chat.yaml

# Step 7.3: Check the Serve application status
serve status

# [Example output]
# name: default
# app_status:
#   status: RUNNING
#   message: ''
#   deployment_timestamp: 1686006910.9571936
# deployment_statuses:
# - name: default_mosaicml--mpt-7b-chat
#   status: HEALTHY
#   message: ''
# - name: default_RouterDeployment
#   status: HEALTHY
#   message: ''

# Step 7.4: List all models
export AVIARY_URL="http://localhost:8000"
aviary models

# [Example output]
# Connecting to Aviary backend at:  http://localhost:8000/
# mosaicml/mpt-7b-chat

# Step 7.5: Send a query to `mosaicml/mpt-7b-chat`.
aviary query --model mosaicml/mpt-7b-chat --prompt "What are the top 5 most popular programming languages?"

# [Example output]
# Connecting to Aviary backend at:  http://localhost:8000/
# mosaicml/mpt-7b-chat:
# 1. Python
# 2. Java
# 3. JavaScript
# 4. C++
# 5. C#
```

## Step 8: Clean up resources

**Warning: GPU nodes are extremely expensive. Please remember to delete the cluster if you no longer need it.**

```sh
# Step 8.1: Delete the RayCluster
# path: deploy/kuberay
kubectl apply -f kuberay.yaml

# Step 8.2: Uninstall the KubeRay operator chart
helm uninstall kuberay-operator

# Step 8.3: Delete the Amazon EKS cluster via AWS Web UI
```
