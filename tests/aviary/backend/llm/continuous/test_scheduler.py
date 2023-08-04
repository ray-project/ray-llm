import asyncio
import os
from typing import List, Optional
from unittest.mock import patch

import pytest

from aviary.backend.llm.continuous.policy import (
    TaskSelectionPolicy,
)
from aviary.backend.llm.continuous.scheduler import InferenceScheduler
from aviary.backend.llm.continuous.tokenizer import TransformersTokenizer
from aviary.backend.llm.continuous.tokenstream import FinishReason
from aviary.backend.llm.continuous.types import InferenceTask, Request, TGIParams
from aviary.backend.server.models import (
    GenerationConfig,
    SchedulerConfig,
    SchedulerPolicyConfig,
    TextGenerationInferenceEngineConfig,
)

DUMMY_VALUE = 1.0


class DummySchedulingPolicy(TaskSelectionPolicy):
    """A dummy scheduling policy that just adds all tasks in the queue to the batch."""

    def select_new_tasks(
        self,
        in_process_tasks: List[InferenceTask],
        queue: asyncio.Queue,
        iterations_since_last_new_batch: Optional[int],
        requires_padding: bool = False,
    ) -> List[InferenceTask]:
        new_tasks = []
        while not queue.empty():
            task = queue.get_nowait()
            new_tasks.append(task)
        return new_tasks

    def on_task_finished(self, finished_tasks: InferenceTask):
        pass

    @property
    def max_input_length(self):
        return 100  # some generic dummy value


@pytest.fixture
def test_worker():
    """Construct a TGI worker for testing."""
    model_id = "hf-internal-testing/tiny-random-gpt2"
    generation_config = GenerationConfig(
        prompt_format={
            "system": "{instruction}",
            "assistant": "{instruction}",
            "trailing_assistant": "",
            "user": "{instruction}",
            "default_system_message": "",
        },
        generate_kwargs={
            "do_sample": True,
            "max_new_tokens": 32,
            "min_new_tokens": 16,
        },
        stopping_sequences=[],
    )

    scheduler_config = SchedulerConfig(policy=SchedulerPolicyConfig(type="dummy"))
    engine_config = TextGenerationInferenceEngineConfig(
        type="",
        model_id=model_id,
        generation=generation_config,
        scheduler=scheduler_config,
    )
    with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
        from aviary.backend.llm.engine.tgi import AviaryTGIInferenceWorker

        worker = AviaryTGIInferenceWorker(engine_config, 1)
        worker.init_model(local_rank=0, num_gpus_per_worker=0, num_cpus_per_worker=1)
        yield worker


@pytest.mark.asyncio
async def test_inference_scheduler(test_worker):
    """Test basic scheduling and running tasks for the InferenceScheduler.

    Check that given 2 tasks where the expected generation length is 3,
    the scheduler runs both tasks 2 completion and returns the expected
    number of tokens.

    """
    EXPECTED_LENGTH = 3

    policy_config = SchedulerPolicyConfig(type="dummy")

    worker = test_worker

    task_selection_policy = DummySchedulingPolicy(
        **policy_config.dict(exclude={"type"})
    )
    inference_task_cls = worker.get_inference_task_cls()
    scheduler = InferenceScheduler(
        inference_worker=worker,
        task_selection_policy=task_selection_policy,
        task_queue=asyncio.Queue(),
        inline=True,
    )

    tokenizer = TransformersTokenizer(worker.get_tokenizer())

    prompt = "a b "  # 3 tokens
    inference_task1 = inference_task_cls(
        Request(
            inputs=prompt,
            input_tokens=tokenizer.get_input_length(prompt),
            truncate=task_selection_policy.max_input_length,
            max_new_tokens=EXPECTED_LENGTH,
            params=TGIParams(min_new_tokens=EXPECTED_LENGTH),
        )
    )
    prompt = "c d e f g"  # 5 tokens
    inference_task2 = inference_task_cls(
        Request(
            inputs=prompt,
            input_tokens=tokenizer.get_input_length(prompt),
            truncate=task_selection_policy.max_input_length,
            max_new_tokens=EXPECTED_LENGTH,
            params=TGIParams(min_new_tokens=EXPECTED_LENGTH),
        )
    )

    scheduler.add_task(inference_task1)
    scheduler.add_task(inference_task2)

    batch_id = None
    in_process_tasks = []
    iterations_since_last_new_batch = 1
    while True:
        (
            in_process_tasks,
            iterations_since_last_new_batch,
            batch_id,
        ) = await scheduler.run_tasks(
            batch_id=batch_id,
            in_process_tasks=in_process_tasks,
            iterations_since_last_new_batch=iterations_since_last_new_batch,
        )
        iterations_since_last_new_batch += 1
        if batch_id is None:
            break

    ret_task_1 = []
    ret_task_2 = []
    while True:
        try:
            ret_task_1.extend(
                await asyncio.gather(inference_task1.output_stream.__anext__())
            )
            ret_task_2.extend(
                await asyncio.gather(inference_task2.output_stream.__anext__())
            )
        except StopAsyncIteration:
            break
    assert len(ret_task_1) == EXPECTED_LENGTH
    assert len(ret_task_2) == EXPECTED_LENGTH


@pytest.mark.asyncio
async def test_inference_scheduler_process_generations(test_worker):
    """Test that the scheduler processes generations correctly.

    Specifically that it:
        - writes out finished tasks generated text to their output streams
        - handles cancelled tasks by removing them from the batch/scheduler
        - keeps unfinished tasks in the scheduler
    """
    # Lazy import to avoid initializing torch.cuda
    from text_generation_server.models.types import GeneratedText, Generation

    policy_config = SchedulerPolicyConfig(type="dummy")

    worker = test_worker

    task_selection_policy = DummySchedulingPolicy(
        **policy_config.dict(exclude={"type"})
    )
    inference_task_cls = worker.get_inference_task_cls()
    scheduler = InferenceScheduler(
        inference_worker=worker,
        task_selection_policy=task_selection_policy,
        task_queue=asyncio.Queue(),
        inline=True,
    )
    tokenizer = TransformersTokenizer(worker.get_tokenizer())
    prompt = "a b "  # 3 tokens
    inference_task = inference_task_cls(
        Request(
            inputs=prompt,
            input_tokens=tokenizer.get_input_length(prompt),
            truncate=task_selection_policy.max_input_length,
            max_new_tokens=3,
            params=TGIParams(min_new_tokens=3),
            id=DUMMY_VALUE,
        )
    )
    expected_token_text = "dummy expected text"
    generation = Generation(
        request_id=DUMMY_VALUE,
        token_text=expected_token_text,
        token_id=DUMMY_VALUE,
        token_logprob=DUMMY_VALUE,
        token_is_special=False,
        prefill_tokens=DUMMY_VALUE,
        generated_text=None,
    )
    # Test that an inprocess task is returned as unfinished, and that the generation is
    # written to the output stream since.
    unfinished_tasks, any_task_finished = scheduler._process_generation_result(
        generations=[generation], tasks=[inference_task]
    )
    assert unfinished_tasks[0] == inference_task
    assert not any_task_finished
    text_in_stream = await inference_task.output_stream.__anext__()
    assert text_in_stream == expected_token_text

    # Test that a cancelled task is
    cancelled_inference_task = inference_task_cls(
        Request(
            inputs=prompt,
            input_tokens=tokenizer.get_input_length(prompt),
            truncate=task_selection_policy.max_input_length,
            max_new_tokens=3,
            params=TGIParams(min_new_tokens=3),
            id=DUMMY_VALUE,
        )
    )
    cancelled_inference_task.mark_as_finished(finish_reason=FinishReason.CANCELLED)
    unfinished_tasks, any_task_finished = scheduler._process_generation_result(
        generations=[generation], tasks=[cancelled_inference_task]
    )
    assert not unfinished_tasks
    assert any_task_finished
    assert (
        cancelled_inference_task.output_stream.finish_reason == FinishReason.CANCELLED
    )
    # no data should be written to the cancelled task's output stream
    with pytest.raises(StopAsyncIteration):
        await cancelled_inference_task.output_stream.__anext__()

    # Test that a correctly finished task is removed from the scheduler
    finish_reason_stopped = 1
    generated_text = GeneratedText(
        text=expected_token_text,
        generated_tokens=1,
        finish_reason=finish_reason_stopped,
        seed=None,
    )
    finished_generation = Generation(
        request_id=DUMMY_VALUE,
        token_text=expected_token_text,
        token_id=DUMMY_VALUE,
        token_logprob=DUMMY_VALUE,
        token_is_special=False,
        prefill_tokens=DUMMY_VALUE,
        generated_text=generated_text,
    )
    unfinished_tasks, any_task_finished = scheduler._process_generation_result(
        generations=[finished_generation], tasks=[inference_task]
    )
    assert not unfinished_tasks
    assert any_task_finished
    assert inference_task.output_stream.finish_reason == FinishReason.STOP


@pytest.mark.asyncio
async def test_inference_scheduler_process_new_tasks(test_worker):
    """Test that the scheduler is running the prefill stage on new tasks properly.

    Specifically check:
        - if new tasks are passed that prefill is run on them
        - if a new task is passed and after prefill they are done, that they are not
            scheduled for generation afterwards.

    """
    policy_config = SchedulerPolicyConfig(type="dummy")
    worker = test_worker
    task_selection_policy = DummySchedulingPolicy(
        **policy_config.dict(exclude={"type"})
    )
    inference_task_cls = worker.get_inference_task_cls()
    scheduler = InferenceScheduler(
        inference_worker=worker,
        task_selection_policy=task_selection_policy,
        task_queue=asyncio.Queue(),
        inline=True,
    )
    tokenizer = TransformersTokenizer(worker.get_tokenizer())
    prompt = "a b "  # 3 tokens
    # check that prefill is completed and inference_task is added to the running batch
    inference_task = inference_task_cls(
        Request(
            inputs=prompt,
            input_tokens=tokenizer.get_input_length(prompt),
            truncate=task_selection_policy.max_input_length,
            max_new_tokens=3,
            params=TGIParams(min_new_tokens=3),
            id=int(DUMMY_VALUE),
        )
    )
    batch_id, batch = await scheduler._process_new_tasks([inference_task])
    assert batch_id is not None, (
        "batch_id should not be None meaning that the batch "
        "was completed after prefill"
    )
    assert batch[0] == inference_task
    generated_token_after_prefill = await batch[0].output_stream.__anext__()
    assert len(generated_token_after_prefill) > 0

    # check that after prefill on a task that finished after prefill, that the task is
    # not scheduled for generation afterwards

    num_tokens_to_generate = 1
    inference_task = inference_task_cls(
        Request(
            inputs=prompt,
            input_tokens=tokenizer.get_input_length(prompt),
            truncate=task_selection_policy.max_input_length,
            max_new_tokens=num_tokens_to_generate,
            params=TGIParams(min_new_tokens=num_tokens_to_generate),
            id=int(DUMMY_VALUE),
        )
    )
    batch_id, batch = await scheduler._process_new_tasks([inference_task])
    assert batch_id is None, (
        "batch_id should be None meaning that the batch was " "completed after prefill"
    )
    assert len(batch) == 0, (
        "batch should be empty meaning that the task was " "completed after prefill"
    )
    generated_token_after_prefill = await inference_task.output_stream.__anext__()
    assert len(generated_token_after_prefill) > 0


@pytest.mark.asyncio
async def test_inference_scheduler_generate_next_token(test_worker):
    """Test that the scheduler is generating tokens properly.

    Specifically check:
        - if a task is passed that has not been prefilled, that it is prefilled
        - if a task is passed that has been prefilled, that it is scheduled for
            generation
        - if a task is passed that has been prefilled and is done, that it is not
            scheduled for generation
    """
    policy_config = SchedulerPolicyConfig(type="dummy")
    worker = test_worker
    task_selection_policy = DummySchedulingPolicy(
        **policy_config.dict(exclude={"type"})
    )
    inference_task_cls = worker.get_inference_task_cls()
    scheduler = InferenceScheduler(
        inference_worker=worker,
        task_selection_policy=task_selection_policy,
        task_queue=asyncio.Queue(),
        inline=True,
    )

    num_tokens_to_generate = 3
    tokenizer = TransformersTokenizer(worker.get_tokenizer())
    prompt = "a b "  # 3 tokens
    # check that prefill is completed and inference_task is added to the running batch
    inference_task = inference_task_cls(
        Request(
            inputs=prompt,
            input_tokens=tokenizer.get_input_length(prompt),
            truncate=task_selection_policy.max_input_length,
            max_new_tokens=num_tokens_to_generate,
            params=TGIParams(min_new_tokens=num_tokens_to_generate),
            id=int(DUMMY_VALUE),
        )
    )
    # 1 token generated
    batch_id, batch = await scheduler._process_new_tasks([inference_task])
    assert batch_id is not None, (
        "batch_id should not be None meaning that the batch "
        "was completed after prefill"
    )
    # 2 tokens generated
    batch_id, batch = await scheduler._generate_next_token([batch_id], tasks=batch)

    assert batch_id is not None
    assert len(batch) == 1

    # 3 tokens generated, the task should be done
    batch_id, batch = await scheduler._generate_next_token([batch_id], tasks=batch)

    assert batch_id is None, "the task should be done therefore batch_id should be None"
    assert len(batch) == 0, "the task should be done therefore batch should be empty"
