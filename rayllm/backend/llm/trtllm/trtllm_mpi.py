from rayllm.backend.llm.trtllm.trtllm_models import TRTLLMGPTServeConfig


def create_server(server_config: TRTLLMGPTServeConfig = None):
    """Start trtllm server with MPI.

    Rank0 process will broadcast the serve config to
    all other ranks processes.
    """
    # Setup CUDA_VISIBILE_DEVICE
    from ray.runtime_env import mpi_init

    mpi_init()

    from mpi4py import MPI
    from tensorrt_llm.libs import trt_llm_engine_py

    comm = MPI.COMM_WORLD
    myrank = comm.Get_rank()
    if myrank == 0:
        assert server_config is not None
        server_config = server_config.dict()

    server_config_dict = comm.bcast(server_config, root=0)

    try:
        server = trt_llm_engine_py.GptServer(**server_config_dict)
    except Exception as e:
        raise e
    if comm.Get_rank() > 0:
        server.wait()
    else:
        return server


if __name__ == "__main__":
    create_server()
