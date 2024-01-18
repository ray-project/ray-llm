import gc
import hashlib
import logging
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from transformers import AutoModel

from rayllm.backend.llm.embedding.embedding_models import (
    EmbeddingApp,
    EmbeddingOptimize,
)

logger = logging.getLogger(__name__)


class EmbeddingModelRunner:
    def __init__(self, llm_app: EmbeddingApp):
        self.llm_app = llm_app
        self.model = self._get_model()

    def _get_model(self):
        try:
            model = AutoModel.from_pretrained(
                self.llm_app.engine_config.actual_hf_model_id, device_map="auto"
            )
        except ValueError:
            model = AutoModel.from_pretrained(
                self.llm_app.engine_config.actual_hf_model_id
            )
            if torch.cuda.is_available():
                model = model.cuda()
        return model

    def _average_pool(
        self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Average pool

        for each input string, the embedding model will encode it to a 10x1024 tensor at
        the last hidden state, we take the average pool on dim 1 (the one with size 10).
        TODO(Terry) update the average pooling logic under batching setup for better performance

        """
        assert (
            len(list(last_hidden_states.size())) == 3
            and list(last_hidden_states.size())[-1] == 1024
        )
        last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    @torch.inference_mode()
    def run_model(
        self, batch_dict: Dict[str, torch.Tensor], normalization: bool = True
    ) -> torch.Tensor:
        batch_dict = {k: t.to(device=self.model.device) for k, t in batch_dict.items()}

        outputs = self.model(**batch_dict)
        embeddings = self._average_pool(
            outputs.last_hidden_state, batch_dict["attention_mask"]
        )

        # return normalized/original embeddings
        t = F.normalize(embeddings) if normalization else embeddings
        t = t.cpu()
        return t

    def __call__(self, *args, **kwargs):
        return self.run_model(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.run_model(*args, **kwargs)


class ONNXGPUEmbeddingModelRunner(EmbeddingModelRunner):
    def _get_model(self):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        from optimum.onnxruntime import (
            AutoOptimizationConfig,
            ORTModelForFeatureExtraction,
            ORTOptimizer,
        )
        from transformers.utils.hub import TRANSFORMERS_CACHE

        class ORTModelForFeatureExtractionPatched(ORTModelForFeatureExtraction):
            # Optimum v0.16.0 introduced an issue where a sentence-transformer model
            # will have buffers named following sentence-transformer convention,
            # but ORTModelForFeatureExtraction requires buffers named following
            # Transformers convention. This patch fixes that.
            def prepare_io_binding(self, *model_inputs, ordered_input_names):
                # Remove input tensors not supported by sentence_transformers
                model_inputs = model_inputs[: len(ordered_input_names)]
                io_binding, output_shapes, output_buffers = super().prepare_io_binding(
                    *model_inputs, ordered_input_names=ordered_input_names
                )
                # Fix output buffers
                if (
                    "last_hidden_state" not in output_buffers
                    and "token_embeddings" in output_buffers
                ):
                    output_buffers["last_hidden_state"] = output_buffers[
                        "token_embeddings"
                    ]
                    output_shapes["last_hidden_state"] = output_shapes[
                        "token_embeddings"
                    ]
                return io_binding, output_shapes, output_buffers

        potential_path = Path(self.llm_app.engine_config.actual_hf_model_id)
        if potential_path.exists():
            # We are dealing with a local path
            has_ort_config = (potential_path / "ort_config.json").exists()
        else:
            has_ort_config = False

        if not has_ort_config:
            path = (
                Path(TRANSFORMERS_CACHE).parent.parent
                / "rayllm_ort_cache"
                / str(
                    hashlib.md5(
                        self.llm_app.engine_config.actual_hf_model_id.encode("utf-8")
                    ).hexdigest()
                )
            )

            if not (path / "ort_config.json").exists():
                # We need to create the model
                logger.info(
                    f"Optimizing model {self.llm_app.engine_config.actual_hf_model_id} with ONNX Runtime..."
                )
                ort_model = ORTModelForFeatureExtractionPatched.from_pretrained(
                    self.llm_app.engine_config.actual_hf_model_id,
                    export=True,
                    provider="CUDAExecutionProvider",
                )
                optimizer = ORTOptimizer.from_pretrained(ort_model)
                optimization_config = AutoOptimizationConfig.O4(for_gpu=True)
                optimizer.optimize(
                    save_dir=str(path), optimization_config=optimization_config
                )
                gc.collect()
        else:
            path = potential_path

        logger.info(f"Loading ORT model from {path}...")
        model = ORTModelForFeatureExtractionPatched.from_pretrained(
            str(path), provider="CUDAExecutionProvider"
        )

        return model


def get_model_runner(llm_app: EmbeddingApp) -> EmbeddingModelRunner:
    if llm_app.engine_config.optimize == EmbeddingOptimize.ONNX:
        return ONNXGPUEmbeddingModelRunner(llm_app)
    return EmbeddingModelRunner(llm_app)
