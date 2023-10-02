# Deploy Aviary on Googke Kubernetes Engine (GKE) using KubeRay

In this tutorial, we will:

1. Set up a Kubernetes cluster on GKE.
2. Deploy the KubeRay operator and a Ray cluster on GKE.
3. Run an LLM model with Aviary.

* Note that this document will be extended to include Ray autoscaling and the deployment of multiple models in the near future.

## Step 1: Create a Kubernetes cluster on GKE

Run this command and all following commands on your local machine or on the [Google Cloud Shell](https://cloud.google.com/shell). If running from your local machine, you will need to install the [Google Cloud SDK](https://cloud.google.com/sdk/docs/install).

```sh
gcloud container clusters create aviary-gpu-cluster \
    --num-nodes=1 --min-nodes 0 --max-nodes 1 --enable-autoscaling \
    --zone=us-west1-b --machine-type e2-standard-8
```

This command creates a Kubernetes cluster named `aviary-gpu-cluster` with 1 node in the `us-west1-b` zone. In this example, we use the `e2-standard-8` machine type, which has 8 vCPUs and 32 GB RAM. The cluster has autoscaling enabled, so the number of nodes can increase or decrease based on the workload.

You can also create a cluster from the [Google Cloud Console](https://console.cloud.google.com/kubernetes/list).

## Step 2: Create a GPU node pool

Run the following command to create a GPU node pool for Ray GPU workers.
(You can also create it from the Google Cloud Console; see the [GKE documentation](https://cloud.google.com/kubernetes-engine/docs/how-to/node-taints#create_a_node_pool_with_node_taints) for more details.)

```sh
gcloud container node-pools create gpu-node-pool \
  --accelerator type=nvidia-l4-vws,count=4 \
  --zone us-west1-b \
  --cluster aviary-gpu-cluster \
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
gcloud container clusters get-credentials aviary-gpu-cluster --zone us-west1-b
```

For more details, see the [GKE documentation](https://cloud.google.com/kubernetes-engine/docs/how-to/cluster-access-for-kubectl).

## Step 4: Install NVIDIA GPU device drivers

This step is required for GPU support on GKE. See the [GKE documentation](https://cloud.google.com/kubernetes-engine/docs/how-to/gpus#installing_drivers) for more details.

```sh
# Install NVIDIA GPU device driver
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded-latest.yaml

# Verify that your nodes have allocatable GPUs. It may take a few seconds for the GPUs to be allocated. 
kubectl get nodes "-o=custom-columns=NAME:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu"

# Example output:
# NAME                                               GPU
# gke-aviary-gpu-cluster-default-pool-ceb8fe4d-8dqw  <none>
# gke-aviary-gpu-cluster-gpu-node-pool-2f4a373c-8q3q 4
```

### Troubleshooting

If you never see the allocatable GPUs, or if the GPU node pool scaled down to zero nodes, you can still proceed to the next step. The GPU device driver will be installed when the GPU node pool scales up again.

To debug issues with the GPU device driver installation, you can run the following command:

```sh
kubectl get pod -n kube-system
```

After finding the correct pod, you can check the logs with:

```sh
kubectl logs -n kube-system nvidia-driver-installer-xxxxx -c nvidia-driver-installer
```

## Step 5: Install the KubeRay operator

```sh
# Install both CRDs and KubeRay operator v0.6.0.
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo update
helm install kuberay-operator kuberay/kuberay-operator --version 0.6.0

# Check that it is scheduled on the CPU node. If it is not, something is wrong.
kubectl get pods -o wide
# Example output:
# NAME                                READY   STATUS    RESTARTS   AGE   IP           NODE                                                NOMINATED NODE   READINESS GATES
# kuberay-operator-54f657c8cf-6ln5j   1/1     Running   0          66s   10.32.0.12   gke-aviary-gpu-cluster-default-pool-ceb8fe4d-8dqw   <none>           <none>
```

## Step 6: Deploy Aviary

At this point, you have two options:

1. You can deploy Aviary manually on a `RayCluster`, or
2. You can deploy Aviary using a [`RayService` custom resource](https://ray-project.github.io/kuberay/guidance/rayservice/).

The first option is more flexible for conducting experiments.  The second option is recommended for production use due to the additional high availability features provided by the `RayService` custom resource, which will manage the underlying `RayCluster`s for you.

If you are running this tutorial on the Google Cloud Shell, please copy the file `docs/kuberay/ray-cluster.aviary-gke.yaml` or `docs/kuberay/ray-service.aviary-gke.yaml` to the Google Cloud Shell, depending on which option you're using. You may find it useful to use the [Cloud Shell Editor](https://cloud.google.com/shell/docs/editor-overview) to edit the file.

Now you can create a RayCluster with Aviary. Aviary is included in the image `anyscale/aviary:latest`, which is specified in the RayCluster YAML manifest `ray-cluster.aviary-gke.yaml`.

Run one of the following two commands:

```sh
# path: docs/kuberay
# Option 1: Deploy Aviary on a RayCluster 
kubectl apply -f ray-cluster.aviary-gke.yaml
```

```sh
# path: docs/kuberay
# Option 2: Deploy Aviary as a RayService
kubectl apply -f ray-service.aviary-gke.yaml
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

* (If using Option 2: Deploy Aviary as a RayService) The `ray-service.aviary-gke.yaml` manifest contains the following Ray Serve options:

    ```yaml
    serviceUnhealthySecondThreshold: 1200 # Config for the health check threshold for service. Default value is 60.
    deploymentUnhealthySecondThreshold: 1200 # Config for the health check threshold for deployments. Default value is 60.
    serveConfigV2: |
        applications:
        - name: amazon--LightGPT
          import_path: aviary.backend:llm_application
          route_prefix: /amazon--LightGPT
          args:
            model: "./models/continuous_batching/amazon--LightGPT.yaml"
    ```

    It also has a field `RayClusterSpec`, which describes the spec for the underlying `RayCluster`. Here we have used the same configuration as in `ray-cluster.aviary-gke.yaml` above, with the following change:
  * We have specified the `containerPort: 8000` with the name `serve` in the head pod spec and the worker pod spec.

## Step 7: Deploy an LLM model with Aviary


In [Step 6](#step-6-deploy-aviary), if you used "Option 1: Deploy Aviary on a RayCluster", please follow [Step 7A](#step-7a-deploy-an-llm-model-with-aviary-on-a-raycluster).  Otherwise, if you used "Option 2: Deploy Aviary as a RayService", please follow [Step 7B](#step-7b-deploy-an-llm-model-with-aviary-as-a-rayservice).

### Step 7A: Deploy an LLM model with Aviary via SSH on a RayCluster



```sh
# Step 7A.1: Log in to the head Pod
export HEAD_POD=$(kubectl get pods --selector=ray.io/node-type=head -o custom-columns=POD:metadata.name --no-headers)
kubectl exec -it $HEAD_POD -- bash

# Step 7A.2: Deploy the `mosaicml/mpt-7b-chat` model
aviary run --model ./models/static_batching/mosaicml--mpt-7b-chat.yaml

# Step 7A.3: Check the Serve application status
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

# Step 7A.4: List all models
export AVIARY_URL="http://localhost:8000"
aviary models

# [Example output]
# Connecting to Aviary backend at:  http://localhost:8000/
# mosaicml/mpt-7b-chat

# Step 7A.5: Send a query to `mosaicml/mpt-7b-chat`.
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

### Step 7B: Deploy an LLM model with Aviary as a RayService

```sh
# Step 7B.0: Wait for the service to be ready.
# Note that the service will be created only when all serve applications are ready. 
kubectl get svc # Check if `aviary-serve-svc` is created.

# If the service is not yet ready, check the status by running `serve status` on the head pod. After a few minutes, the status should move from UPDATING to HEALTHY.
# You can find the head pod by running `kubectl get pod`.

# kubectl exec -it aviary-raycluster-sfz6r-head-59g8h -- serve status

# name: amazon--LightGPT
# app_status:
#   status: DEPLOYING
#   message: ''
#   deployment_timestamp: 1691448736.5297794
# deployment_statuses:
# - name: amazon--LightGPT_amazon--LightGPT
#   status: UPDATING
#   message: Deployment amazon--LightGPT_amazon--LightGPT has 1 replicas that have taken
#     more than 30s to initialize. This may be caused by a slow __init__ or reconfigure
#     method.

# Step 7B.1: Port forward the Kubernetes serve service.  This command will block, so please open a new terminal.
kubectl port-forward service/aviary-serve-svc 8000:8000

# Step 7B.2: Ensure the Aviary client is installed. You may also need to run `pip install boto3` and `pip install pydantic` if you run into Python import errors.
pip install "aviary @ git+https://github.com/ray-project/aviary.git"

# Step 7B.3: List models via the Aviary CLI outside the Kubernetes cluster.
export AVIARY_URL="http://localhost:8000"
aviary models

# Example output:
# Connecting to Aviary backend at: http://localhost:8000/v1
# amazon/LightGPT

# Step 7B.4: Send a query to `amazon/LightGPT`.
aviary query --model amazon/LightGPT --prompt "What are the top 5 most popular programming languages?"

# Example output:
# Connecting to Aviary backend at: http://localhost:8000/v1
# amazon/LightGPT:
# 1. JavaScript
# 2. Java
# 3. Python
# 4. C++
# 5. C#
```

## Step 8: Clean up resources

**Warning: GPU nodes are extremely expensive. Please remember to delete the cluster if you no longer need it.**

Run one of the following commands, depending on which option you chose in [Step 6](#step-6-deploy-aviary).

```sh
# Step 8.1A: Delete the RayCluster custom resource
# path: docs/kuberay
kubectl delete -f ray-cluster.aviary-gke.yaml
```

```sh
# Step 8.1B: Delete the RayService custom resource
# path: docs/kuberay
kubectl delete -f ray-service.aviary-gke.yaml
```

Finally, run the following commands to delete the KubeRay operator chart and the GKE cluster.

```sh

# Step 8.2: Uninstall the KubeRay operator chart
helm uninstall kuberay-operator

# Step 8.3: Delete the GKE cluster
gcloud container clusters delete aviary-gpu-cluster
```

See the [GKE documentation](https://cloud.google.com/kubernetes-engine/docs/how-to/deleting-a-cluster) for more details on deleting a GKE cluster.
