import asyncio
import time
import traceback
import uuid
from typing import Any, Dict, List, Optional, Union

import async_timeout
import ray
import ray.util
from fastapi import FastAPI, Request
from ray import serve
from ray.exceptions import RayActorError
from ray.serve.deployment import ClassNode

from aviary.backend.llm.predictor import LLMPredictor
from aviary.backend.logger import get_logger
from aviary.backend.server._batch import QueuePriority, _PriorityBatchQueue, batch
from aviary.backend.server.exceptions import PromptTooLongError
from aviary.backend.server.models import (
    Args,
    DeepSpeed,
    Prompt,
)
from aviary.common.constants import GATEWAY_TIMEOUT_S
from aviary.common.models import (
    Model,
    ModelData,
    Completion,
    ChatCompletion,
    Usage,
    TextChoice,
    Message,
    MessageChoices
)

logger = get_logger(__name__)


@serve.deployment(
    autoscaling_config={
        "min_replicas": 1,
        "initial_replicas": 2,
        "max_replicas": 8,
    },
    max_concurrent_queries=2,  # Maximum backlog for a single replica
    health_check_period_s=10,
    health_check_timeout_s=30,
)
class LLMDeployment(LLMPredictor):
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.args = None
        super().__init__()

    def _should_reinit_worker_group(self, new_args: Args) -> bool:
        old_args = self.args

        if not old_args:
            return True

        old_scaling_config = self.args.air_scaling_config
        new_scaling_config = new_args.scaling_config.as_air_scaling_config()

        if not self.base_worker_group:
            return True

        if old_scaling_config != new_scaling_config:
            return True

        if not old_args:
            return True

        if old_args.model_config.initialization != new_args.model_config.initialization:
            return True

        if (
            old_args.model_config.generation.max_batch_size
            != new_args.model_config.generation.max_batch_size
            and isinstance(new_args.model_config.initialization.initializer, DeepSpeed)
        ):
            return True

        # TODO: Allow this
        if (
            old_args.model_config.generation.prompt_format
            != new_args.model_config.generation.prompt_format
        ):
            return True

        return False

    async def reconfigure(
        self,
        config: Union[Dict[str, Any], Args],
        force: bool = False,
    ) -> None:
        logger.info("Reconfiguring...")
        if not isinstance(config, Args):
            new_args: Args = Args.parse_obj(config)
        else:
            new_args: Args = config

        should_reinit_worker_group = force or self._should_reinit_worker_group(new_args)

        self.args = new_args
        if should_reinit_worker_group:
            await self.rollover(
                self.args.air_scaling_config,
                pg_timeout_s=self.args.scaling_config.pg_timeout_s,
            )
        logger.info("Reconfigured.")

    @property
    def max_batch_size(self):
        return self.args.model_config.generation.max_batch_size

    @property
    def batch_wait_timeout_s(self):
        return self.args.model_config.generation.batch_wait_timeout_s

    def get_max_batch_size(self):
        return self.max_batch_size

    def get_batch_wait_timeout_s(self):
        return self.batch_wait_timeout_s

    async def validate_prompt(self, prompt: Prompt) -> None:
        if len(prompt.prompt.split()) > self.args.model_config.max_input_words:
            raise PromptTooLongError(
                f"Prompt exceeds max input words of "
                f"{self.args.model_config.max_input_words}. "
                "Please make the prompt shorter."
            )

    async def metadata(self) -> dict:
        return self.args.dict(
            exclude={
                "model_config": {"initialization": {"s3_mirror_config", "runtime_env"}}
            }
        )

    async def generate_text(self, prompt: Prompt):
        await self.validate_prompt(prompt)
        time.time()
        with async_timeout.timeout(GATEWAY_TIMEOUT_S):
            text = await self.generate_text_batch(
                prompt,
                priority=QueuePriority.GENERATE_TEXT,
                # start_timestamp=start_timestamp,
            )
            return text

    async def batch_generate_text(self, prompts: List[Prompt]):
        for prompt in prompts:
            await self.validate_prompt(prompt)
        time.time()
        with async_timeout.timeout(GATEWAY_TIMEOUT_S):
            texts = await asyncio.gather(
                *[
                    self.generate_text_batch(
                        prompt,
                        priority=QueuePriority.BATCH_GENERATE_TEXT,
                        # start_timestamp=start_timestamp,
                    )
                    for prompt in prompts
                ]
            )
            return texts

    @batch(
        max_batch_size=get_max_batch_size,
        batch_wait_timeout_s=get_batch_wait_timeout_s,
        batch_queue_cls=_PriorityBatchQueue,
    )
    async def generate_text_batch(
        self,
        prompts: List[Prompt],
        *,
        start_timestamp: Optional[Union[float, List[float]]] = None,
        timeout_s: Union[float, List[float]] = GATEWAY_TIMEOUT_S - 10,
    ):
        """Generate text from the given prompts in batch.

        Args:
            prompts (List[Prompt]): Batch of prompts to generate text from.
            start_timestamp (Optional[float], optional): Timestamp of when the
                batch was created. Defaults to None. If set, will early stop
                the generation.
            timeout_s (float, optional): Timeout for the generation. Defaults
                to GATEWAY_TIMEOUT_S-10. Ignored if start_timestamp is None.
        """
        if not prompts or prompts[0] is None:
            return prompts

        if isinstance(start_timestamp, list) and start_timestamp[0]:
            start_timestamp = min(start_timestamp)
        elif isinstance(start_timestamp, list):
            start_timestamp = start_timestamp[0]
        if isinstance(timeout_s, list) and timeout_s[0]:
            timeout_s = min(timeout_s)
        elif isinstance(timeout_s, list):
            timeout_s = timeout_s[0]

        logger.info(
            f"Received {len(prompts)} prompts {prompts}. start_timestamp {start_timestamp} timeout_s {timeout_s}"
        )
        data_ref = ray.put(prompts)

        while not self.base_worker_group:
            logger.info("Waiting for worker group to be initialized...")
            await asyncio.sleep(1)

        try:
            prediction = await self._predict_async(
                data_ref, timeout_s=timeout_s, start_timestamp=start_timestamp
            )
        except RayActorError as e:
            raise RuntimeError(
                f"Prediction failed due to RayActorError. "
                "This usually means that one or all prediction workers are dead. "
                "Try again in a few minutes. "
                f"Traceback:\n{traceback.print_exc()}"
            ) from e

        logger.info(f"Predictions {prediction}")
        if not isinstance(prediction, list):
            return [prediction]
        return prediction[: len(prompts)]

    # Called by Serve to check the replica's health.
    async def check_health(self):
        if self._new_worker_group_lock.locked():
            logger.info("Rollover in progress, skipping health check")
            return
        if self.pg and self.base_worker_group:
            dead_actors = []
            for actor in self.base_worker_group:
                actor_state = ray.state.actors(actor._ray_actor_id.hex())
                if actor_state["State"] == "DEAD":
                    dead_actors.append(actor)
            if dead_actors:
                raise RuntimeError(
                    f"At least one prediction worker is dead. Dead workers: {dead_actors}. "
                    "Reinitializing worker group."
                )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}:{self.args.model_config.model_id}"


def _replace_prefix(model: str) -> str:
    return model.replace("--", "/")


class ExecutionHooks:
    def __init__(self):
        self.hooks = []

    def add_post_execution_hook(self, fn):
        self.hooks.append(fn)

    async def trigger_post_execution_hook(
        self, request: Request, model_id: str, input_str: str, output_str: str
    ):
        # Run the token hooks in parallel
        # If a token hook fails, the request will fail
        assert len(self.hooks) > 0, "There should be at least 1 token hook."
        await asyncio.gather(
            *[fn(request, model_id, input_str, output_str) for fn in self.hooks]
        )


app = FastAPI()


class Router:
    def __init__(
        self,
        models: Dict[str, ClassNode],
        model_configurations: Dict[str, Args],
        hooks=None,
    ) -> None:
        self._models = models
        # TODO: Remove this once it is possible to reconfigure models on the fly
        self._model_configurations = model_configurations
        self.hooks = hooks or ExecutionHooks()

    @app.post("/query/{model}")
    async def query(
        self, model: str, prompt: Prompt, request: Request
    ) -> Dict[str, Dict[str, Any]]:
        model = _replace_prefix(model)
        results = await asyncio.gather(
            *(await asyncio.gather(*[self._models[model].generate_text.remote(prompt)]))
        )
        results = results[0]
        logger.info(results)
        print("Query results", results)

        # Set execution state on the request object for middlewares
        await self.hooks.trigger_post_execution_hook(
            request, model, prompt.prompt, results
        )

        return {model: results}

    @app.post("/query/batch/{model}")
    async def batch_query(
        self, model: str, prompts: List[Prompt]
    ) -> Dict[str, List[Dict[str, Any]]]:
        model = _replace_prefix(model)
        results = await asyncio.gather(
            *(
                await asyncio.gather(
                    *[self._models[model].batch_generate_text.remote(prompts)]
                )
            )
        )
        results = results[0]
        logger.info(results)
        return {model: results}

    @app.get("/metadata/{model}")
    async def metadata(self, model) -> Dict[str, Dict[str, Any]]:
        model = _replace_prefix(model)
        # This is what we want to do eventually, but it looks like reconfigure
        # is blocking when called on replica init
        # metadata = await asyncio.gather(
        #     *(await asyncio.gather(*[self._models[model].metadata.remote()]))
        # )
        # metadata = metadata[0]
        metadata = self._model_configurations[model].dict(
            exclude={
                "model_config": {"initialization": {"s3_mirror_config", "runtime_env"}}
            }
        )
        logger.info(metadata)
        return {"metadata": metadata}

    @app.get("/models")
    async def models(self) -> List[str]:
        return list(self._models.keys())

    @app.get("/v1/models", response_model=Model)
    async def models(self) -> Model:
        """OpenAI API-compliant endpoint to get all Aviary models."""
        model_ids = list(self._models.keys())
        model_data = []
        for model_id in model_ids:
            model_data.append(ModelData(
                id=model_id,
                object="model",
                owned_by="organization-owner",  # TODO: define owner (metadata)
                permission=[]  # TODO: define permissions (metadata)
            ))
        return Model(data=model_data)

    @app.get("/v1/models/{model}", response_model=ModelData)
    async def models(self, model: str) -> ModelData:
        """OpenAI API-compliant endpoint to get one Aviary model.

        :param model: The Aviary model ID (e.g. "amazon/LightGPT")
        """
        # TODO: should we integrate "metadata" here?
        return ModelData(
            id=model,
            object="model",
            owned_by="organization-owner",  # TODO
            permission=[]  # TODO
        )

    @app.post("/v1/completions/{model}", response_model=Completion)
    async def completions(
            self,
            model: str,
            prompt: Union[Prompt, List[Prompt]] = Prompt(text="<|endoftext|>"),
            suffix: str = None,
            max_tokens: int = 32,
            temperature: float = 1.0,
            top_p: float = 1.0,
            n: int = 1,
            stream: bool = False,
            logprobs: int = None,
            echo: bool = False,
            stop: str = None,
            presence_penalty: float = 0.0,
            frequency_penalty: float = 0.0,
            best_of: int = 1,
            logit_bias: Dict[str, float] = None,
            user: str = None,
    ) -> Completion:
        """Given a prompt, the model will return one or more predicted completions,
        and can also return the probabilities of alternative tokens at each position.

        Args:
            model: The model to query.
            prompt: The prompt(s) to generate completions for, encoded as string
                or list of strings.
            suffix: The suffix that comes after a completion of inserted text.
            max_tokens: The maximum number of tokens to generate.
            temperature: What sampling temperature to use.
            top_p: An alternative to sampling with temperature, called nucleus sampling.
            n: How many completions to generate for each prompt.
            stream: Whether to stream back partial progress.
            logprobs: Include the log probabilities on the `logprobs` most likely
                tokens, as well the chosen tokens.
            echo: Echo back the prompt in addition to the completion.
            stop: Up to 4 sequences where the API will stop generating further tokens.
                The returned text will not contain the stop sequence.
            presence_penalty: Number between -2.0 and 2.0.
                Positive values penalize new tokens based on whether they appear in
                the text so far, increasing the model's likelihood to talk about
                new topics.
            frequency_penalty: Number between -2.0 and 2.0. Positive values penalize
                new tokens based on their existing frequency in the text so far,
                decreasing the model's likelihood to repeat the same line verbatim.
            best_of: Generates `best_of` completions server-side and returns the "best".
            logit_bias: Modify the likelihood of specified tokens appearing in
                the completion.
            user: A unique identifier representing your end-user, which can help us
                to monitor and detect abuse. Learn more.

        Returns:
            A response object with completions.
        """
        model = _replace_prefix(model)
        results = await asyncio.gather(
            *(await asyncio.gather(*[self._models[model].generate_text.remote(prompt)]))
        )
        results = results[0]
        logger.info(results)

        choices = [TextChoice(
            text=results["generated_text"],
            index=0,
            logprobs={},
            finish_reason="length"
        )]
        usage = Usage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        )
        # TODO: pick up parameters that make sense, remove the rest

        return Completion(
            id=model + "-" + str(uuid.uuid4()),
            object="text_completion",
            created=int(time.time()),
            model=model,
            choices=choices,
            usage=usage,
        )

    @app.post("/v1/chat/completions/{model}", response_model=ChatCompletion)
    async def chat(
            self,
            model: str,
            messages: List[Message],
            temperature: float = 1.0,
            top_p: float = 1.0,
            n: int = 1,
            stream: bool = False,
            logprobs: int = None,
            echo: bool = False,
            stop: str = None,
            presence_penalty: float = 0.0,
            frequency_penalty: float = 0.0,
            logit_bias: Dict[str, float] = None,
            user: str = None,
    ) -> ChatCompletion:
        """Given a prompt, the model will return one or more predicted completions,
        and can also return the probabilities of alternative tokens at each position.

        Args:
            model: The model to query.
            messages: A list of messages describing the conversation so far.
                Contains a required "role", which is the role of the author of this
                message. One of "system", "user", or "assistant".
                Also contains required "content", the contents of the message, and
                an optional "name", the name of the author of this message.
            temperature: What sampling temperature to use.
            top_p: An alternative to sampling with temperature, called nucleus sampling.
            n: How many completions to generate for each prompt.
            stream: Whether to stream back partial progress.
            logprobs: Include the log probabilities on the `logprobs` most likely
                tokens, as well the chosen tokens.
            echo: Echo back the prompt in addition to the completion.
            stop: Up to 4 sequences where the API will stop generating further tokens.
                The returned text will not contain the stop sequence.
            presence_penalty: Number between -2.0 and 2.0.
                Positive values penalize new tokens based on whether they appear in
                the text so far, increasing the model's likelihood to talk about
                new topics.
            frequency_penalty: Number between -2.0 and 2.0. Positive values penalize
                new tokens based on their existing frequency in the text so far,
                decreasing the model's likelihood to repeat the same line verbatim.
            logit_bias: Modify the likelihood of specified tokens appearing in
                the completion.
            user: A unique identifier representing your end-user, which can help us
                to monitor and detect abuse. Learn more.

        Returns:
            A response object with completions.
        """
        model = _replace_prefix(model)
        prompt = messages[-1].content  # FIXME
        results = await asyncio.gather(
            *(await asyncio.gather(
                *[self._models[model].generate_text.remote(prompt)]
            )))
        results = results[0]
        logger.info(results)
        # TODO: pick up parameters that make sense, remove the rest

        choices: List[MessageChoices] = [MessageChoices(
            message=Message(
                role="assistant",
                content=results["generated_text"]),
            index=0,
            finish_reason="length"
        )]
        usage = Usage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        )

        return ChatCompletion(
            id=model + "-" + str(uuid.uuid4()),
            object="text_completion",
            created=int(time.time()),
            model=model,
            choices=choices,
            usage=usage,
        )


RouterDeployment = serve.deployment(
    route_prefix="/",
    # TODO make this configurable in aviary run
    autoscaling_config={
        "min_replicas": 2,
        "initial_replicas": 2,
        "max_replicas": 16,
    },
    max_concurrent_queries=50,  # Maximum backlog for a single replica
)(serve.ingress(app)(Router))
