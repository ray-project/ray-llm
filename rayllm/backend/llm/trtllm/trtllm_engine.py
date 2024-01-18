from typing import TYPE_CHECKING, AsyncIterator, Dict, List, Tuple

from transformers import AutoTokenizer

from rayllm.backend.llm.trtllm.trtllm_models import (
    TRTLLMApp,
    TRTLLMGenerationRequest,
    TRTLLMGPTServeConfig,
)
from rayllm.backend.llm.trtllm.trtllm_mpi import create_server
from rayllm.backend.observability.fn_call_metrics import ClockUnit, MsClock
from rayllm.backend.server.models import AviaryModelResponse
from rayllm.common.utils import download_files_from_s3

if TYPE_CHECKING:
    import tensorrt_llm.libs.trt_llm_engine_py as trt_llm_engine_py
    import torch


class TRTLLMOutput:
    """TRTLLM output wrapper"""

    def __init__(self, stream_handle, tokenizer):
        self._read_offset: int = 0
        self._output_ids: List[int] = []
        self._stream_handle: "trt_llm_engine_py.WorkItem" = stream_handle
        self._tokenizer = tokenizer

        # Only support beam size is 1 and one batch at a time.
        self._input_index = 0
        self._beam_size = 0

    def _get(self) -> Tuple[str, int]:
        """Generate new output text.

        return: new_output_text, num_generated_tokens
        """

        # prefix_offset is used to handle the space generation.
        # Only return the text ouput_ids[read_offset:].
        prefix_offset = max(0, self._read_offset - 5)
        prefix_text = self._tokenizer.decode(
            self._output_ids[prefix_offset : self._read_offset]
        )
        new_text = self._tokenizer.decode(self._output_ids[prefix_offset:])
        if len(new_text) > len(prefix_text):
            new_text = new_text[len(prefix_text) :]
            num_tokens = len(self._output_ids) - self._read_offset
            self._read_offset = len(self._output_ids)
            return new_text, num_tokens
        return None, 0

    def _put(self, tokens: Dict[str, "torch.Tensor"]):  # noqa: F821
        # Pywrapper return a dict from the trtllm engine.
        # "output_ids" is the key to represent the outputs.
        if "output_ids" not in tokens:
            return
        output_ids = tokens["output_ids"].tolist()
        self._output_ids += output_ids[self._input_index][self._beam_size]

    def __aiter__(self):
        return self

    async def __anext__(self):
        # Stop the iteration when stream_output is None.
        stream_output = await self._stream_handle
        if stream_output is None:
            raise StopAsyncIteration
        self._put(stream_output)
        return self._get()


class TRTLLMEngine:
    def __init__(self, config: TRTLLMApp):
        """Initialize the TRTLLM engine

        Limited support: download file from s3/local file path.
        """

        self.engine_config = config.engine_config
        self.tokenizer = AutoTokenizer.from_pretrained(config.engine_config.model_id)
        if config.engine_config.model_local_path:
            path = config.engine_config.model_local_path
        elif (
            config.engine_config.s3_mirror_config
            and config.engine_config.s3_mirror_config.bucket_uri
        ):
            path = f"/tmp/{config.engine_config.model_id}"
            download_files_from_s3(
                config.engine_config.s3_mirror_config.bucket_uri, path
            )
        else:
            raise Exception("No config for TRTLLM model location")

        # Construct the gpt server config
        gpt_serve_config = TRTLLMGPTServeConfig.from_engine_config(
            path, config.engine_config
        )
        self.engine = create_server(gpt_serve_config)
        self.stopping_seq_ids = None
        if config.engine_config.generation.stopping_sequences:
            if isinstance(config.engine_config.generation.stopping_sequences, list):
                # Avoid the tensorrt llm dependency.
                from tensorrt_llm.runtime import to_word_list_format

                stop_words_list = to_word_list_format(
                    [config.engine_config.generation.stopping_sequences], self.tokenizer
                )
                self.stopping_seq_ids = stop_words_list.flatten().tolist()
            else:
                raise ValueError("Only support a list for stopping sequences")

    async def generate(
        self, request: TRTLLMGenerationRequest
    ) -> AsyncIterator[AviaryModelResponse]:
        """Inferece with trtllm engine"""
        kwargs = request.sampling_params.dict()
        input_tokens = self.tokenizer.encode(request.prompt)
        output_len = (
            request.sampling_params.max_tokens or self.engine_config.max_total_tokens
        )
        kwargs.pop("max_tokens", None)

        # TRTLLM engine require int variable as request id.
        request_id = abs(hash(request.request_id))
        try:
            stream = self.engine.enqueue(
                request_id,
                input_tokens,
                output_len,
                request.stream,
                stop_words_list=self.stopping_seq_ids,
                end_id=self.tokenizer.eos_token_id,
                pad_id=self.tokenizer.eos_token_id,
                **kwargs,
            )
            trtllm_output = TRTLLMOutput(stream, self.tokenizer)

            clock = MsClock(unit=ClockUnit.s)
            async for output_text, num_generated_tokens in trtllm_output:
                if num_generated_tokens:
                    yield AviaryModelResponse(
                        generated_text=output_text,
                        num_generated_tokens=num_generated_tokens,
                        num_generated_tokens_batch=1,
                        num_input_tokens_batch=len(input_tokens),
                        generation_time=clock.reset_interval(),
                    )
        finally:
            self.engine.stop(request_id)
