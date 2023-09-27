import ray

ray.kill(ray.get_actor("SERVE_CONTROLLER_ACTOR", namespace="serve"))
