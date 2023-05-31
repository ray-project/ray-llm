import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import deepspeed
import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel

from aviary.backend.logger import get_logger

from .base import TransformersInitializer

logger = get_logger(__name__)


# TODO: Allow deepspeed kwargs
class DeepSpeedInitializer(TransformersInitializer):
    """Initialize model (with DeepSpeed) and tokenizer and place them on the correct device.

    Args:
        device (torch.device): Device to place model and tokenizer on.
        world_size (int): Number of GPUs to use.
        dtype (torch.dtype, optional): Data type to use. Defaults to torch.float16.
        use_bettertransformer (bool, optional): Whether to use BetterTransformer. Defaults to False.
        torch_compile (Optional[Dict[str, Any]], optional): Parameters for ``torch.compile``. Defaults to None.
        max_tokens (int, optional): Maximum number of tokens to use. Defaults to 1024.
        use_kernel (bool, optional): Whether to use the DeepSpeed kernel injection. Defaults to False.
        use_meta_tensor (bool, optional): Whether to use meta tensor loading method. Defaults to False.
        injection_policy ([type], optional): Injection policy for DeepSpeed AutoTP. Cannot
            be set if use_kernel=True. Defaults to None.
        **from_pretrained_kwargs: Keyword arguments for ``AutoModel.from_pretrained``.
    """

    def __init__(
        self,
        device: torch.device,
        world_size: int,
        dtype: torch.dtype = torch.float16,
        use_bettertransformer: bool = False,
        torch_compile: Optional[Dict[str, Any]] = None,
        max_tokens: int = 1024,
        use_kernel: bool = False,
        use_meta_tensor: bool = False,
        injection_policy=None,
        **from_pretrained_kwargs,
    ):
        super().__init__(
            device=device,
            world_size=world_size,
            dtype=dtype,
            use_bettertransformer=use_bettertransformer,
            torch_compile=torch_compile,
            **from_pretrained_kwargs,
        )
        self.max_tokens = max_tokens
        self.use_kernel = use_kernel
        self.use_meta_tensor = use_meta_tensor
        # TODO: Allow conversion from strings (need to do dynamic imports)
        self.injection_policy = injection_policy

        if self.use_kernel:
            assert not (self.use_bettertransformer or self.torch_compile)

        if self.use_meta_tensor:
            assert self.use_kernel

    def _get_model_from_pretrained_kwargs(self):
        return dict(
            low_cpu_mem_usage=True,
            torch_dtype=self.dtype,
            **self.from_pretrained_kwargs,
        )

    # From https://github.com/microsoft/DeepSpeedExamples/blob/master/inference/huggingface/text-generation/utils.py
    def _generate_checkpoint_json(
        self, model_id: str, checkpoint_path: Optional[str] = None
    ) -> Tuple[str, str]:
        if checkpoint_path is None:
            repo_root = snapshot_download(
                model_id,
                allow_patterns=["*"],
                ignore_patterns=["*.safetensors", "*.h5", "*.msgpack"],
                local_files_only=False,
                revision=None,
            )
        else:
            assert os.path.exists(
                checkpoint_path
            ), f"Checkpoint path {checkpoint_path} does not exist"
            repo_root = checkpoint_path

        if os.path.exists(os.path.join(repo_root, "ds_inference_config.json")):
            checkpoints_json = os.path.join(repo_root, "ds_inference_config.json")
        elif model_id in [
            "microsoft/bloom-deepspeed-inference-int8",
            "microsoft/bloom-deepspeed-inference-fp16",
        ]:
            # tp presharded repos come with their own checkpoints config file
            checkpoints_json = os.path.join(repo_root, "ds_inference_config.json")
        else:
            checkpoints_json = os.path.join(repo_root, "checkpoints.json")

            with open(checkpoints_json, "w", encoding="utf-8") as f:
                file_list = [
                    str(entry).split("/")[-1]
                    for entry in Path(repo_root).rglob("*.[bp][it][n]")
                    if entry.is_file()
                ]
                data = {"type": "BLOOM", "checkpoints": file_list, "version": 1.0}
                json.dump(data, f)

        return os.path.abspath(repo_root), os.path.abspath(checkpoints_json)

    def load_model(self, model_id: str) -> "PreTrainedModel":
        model_id_or_path = self._get_model_location_on_disk(model_id)

        logger.info(f"Loading model {model_id_or_path}...")
        if self.use_meta_tensor:
            logger.info("Loading model using DeepSpeed meta tensor...")
            config = AutoConfig.from_pretrained(
                model_id_or_path, **self._get_model_from_pretrained_kwargs()
            )
            self._repo_root, self._checkpoints_json = self._generate_checkpoint_json(
                model_id
            )

            with deepspeed.OnDevice(dtype=torch.float16, device="meta"):
                model = AutoModelForCausalLM.from_config(config)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id_or_path, **self._get_model_from_pretrained_kwargs()
            )
        model.eval()
        return model

    def postprocess_model(self, model: "PreTrainedModel") -> "PreTrainedModel":
        from transformers import GPTNeoXForCausalLM, LlamaForCausalLM

        injection_policy = self.injection_policy
        if injection_policy is None and not self.use_kernel:
            if isinstance(model, GPTNeoXForCausalLM):
                from transformers import GPTNeoXLayer

                injection_policy = {
                    GPTNeoXLayer: ("attention.dense", "mlp.dense_4h_to_h")
                }
            elif isinstance(model, LlamaForCausalLM):
                from transformers.models.llama.modeling_llama import LlamaDecoderLayer

                injection_policy = {
                    LlamaDecoderLayer: ("self_attn.o_proj", "mlp.down_proj")
                }

        if self.use_bettertransformer:
            from optimum.bettertransformer import BetterTransformer

            logger.info("Transforming the model with BetterTransformer...")
            model = BetterTransformer.transform(model)

        if self.use_meta_tensor:
            ds_kwargs = dict(
                base_dir=self._repo_root, checkpoint=self._checkpoints_json
            )
        else:
            ds_kwargs = dict()

        model = deepspeed.init_inference(
            model,
            dtype=self.dtype,
            mp_size=self.world_size,
            replace_with_kernel_inject=self.use_kernel,
            injection_policy=injection_policy,
            max_tokens=self.max_tokens,
            **ds_kwargs,
        )

        if self.torch_compile and self.torch_compile["backend"]:
            logger.info("Compiling the model with torch.compile()...")
            model = torch.compile(model, **self.torch_compile)

        # Add attributes for compatibility with the pipeline
        model.use_kernel = self.use_kernel
        model.device = self.device
        model = model.to(self.device)
        return model
