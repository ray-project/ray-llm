from typing import TYPE_CHECKING

from vllm.worker.worker import Worker

from aviary.backend.llm.vllm.metrics.vllm_worker import (
    model_metrics,
    num_input_tokens_gauge,
    num_seq_groups_gauge,
    sampler_metrics,
    worker_metrics,
)

if TYPE_CHECKING:
    from vllm.model_executor.models.llama import LlamaForCausalLM


def instrument_model(model: "LlamaForCausalLM") -> "LlamaForCausalLM":
    model.model.forward = model_metrics.wrap(model.model.forward)
    model.sampler.forward = sampler_metrics.wrap(model.sampler.forward)
    return model


class InstrumentedWorker(Worker):
    @worker_metrics.wrap
    def init_model(self, *args, **kwargs):
        ret = super().init_model(*args, **kwargs)
        # Postprocess model
        self.model = instrument_model(self.model)
        return ret

    @worker_metrics.wrap
    def _prepare_inputs(self, *args, **kwargs):
        ret = super()._prepare_inputs(*args, **kwargs)
        num_input_tokens_gauge.set(self.num_input_tokens)
        num_seq_groups_gauge.set(self.num_seq_groups)
        return ret

    @worker_metrics.wrap
    def execute_model(self, *args, **kwargs):
        return super().execute_model(*args, **kwargs)
