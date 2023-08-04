import asyncio

import pytest

from aviary.backend.llm.continuous.policy import Quota, QuotaBasedTaskSelectionPolicy
from aviary.backend.llm.continuous.types import InferenceTask, Request
from aviary.backend.llm.engine.tgi import TGIParams


class TestQuotaBasedTaskSelectionPolicy:
    def test_calculate_quota(self):
        """Test that the quota is calculated correctly."""
        policy = QuotaBasedTaskSelectionPolicy(
            max_batch_total_tokens=20,  # Only 1 task can fit
            max_batch_prefill_tokens=10,
            max_input_length=5,
            max_total_tokens=10,
        )

        task1 = InferenceTask(
            Request(
                inputs="a b ",
                input_tokens=5,
                truncate=5,
                max_new_tokens=10,
                params=TGIParams(),
            )
        )

        task2 = InferenceTask(
            Request(
                inputs="c d ",
                input_tokens=5,
                truncate=5,
                max_new_tokens=10,
                params=TGIParams(),
            )
        )

        task3 = InferenceTask(
            Request(
                inputs="e f ",
                input_tokens=3,
                truncate=5,
                max_new_tokens=2,
                params=TGIParams(),
            )
        )

        # test quota for a single task
        in_process_tasks = [task1]
        quota = policy.calculate_quota(in_process_tasks)
        assert quota == Quota(token_budget=5, prefill_token_budget=10)

        # test quota for multiple tasks that won't fit in the budget
        with pytest.raises(ValueError, match="Token budget cannot be negative."):
            in_process_tasks = [task1, task2]
            quota = policy.calculate_quota(in_process_tasks)

        # test quota for multiple tasks that will fit in the budget
        in_process_tasks = [task1, task3]
        quota = policy.calculate_quota(in_process_tasks)
        assert quota == Quota(token_budget=0, prefill_token_budget=10)

    def test_select_new_tasks(self):
        """Test that quota based selection works correctly."""

        policy = QuotaBasedTaskSelectionPolicy(
            max_batch_total_tokens=30,  # 2 tasks can fit
            max_batch_prefill_tokens=10,
            max_input_length=5,
            max_total_tokens=10,
        )

        task1 = InferenceTask(
            Request(
                inputs="a b ",
                input_tokens=5,
                truncate=5,
                max_new_tokens=10,
                params=TGIParams(),
            )
        )

        # Check that adding one task when there are no in_process_tasks returns that task

        queue = asyncio.Queue()
        for _ in range(2):
            queue.put_nowait(task1)
        new_tasks = policy.select_new_tasks(
            in_process_tasks=[], queue=queue, iterations_since_last_new_batch=0
        )
        for task in new_tasks:
            assert task == task1

        # Check that adding when trying to add more tasks than the budget,
        # only the tasks that are in budget are returned

        queue = asyncio.Queue()
        # 2 tasks should fit the budget, but not 3
        for _ in range(3):
            queue.put_nowait(task1)

        new_tasks = policy.select_new_tasks(
            in_process_tasks=[], queue=queue, iterations_since_last_new_batch=0
        )
        assert len(new_tasks) == 2
        for task in new_tasks:
            assert task == task1

        # similarly if there are in_process_tasks, only allow tasks that fit in the
        # adjusted budget to be scheduled

        in_process_tasks = [task1]
        quota = policy.calculate_quota(in_process_tasks)
        # the remaining budget should be 15 since the total budget is 30 and
        # the first task has a total budget of 15 (5 input tokens + 10 max new tokens)
        assert quota == Quota(token_budget=15, prefill_token_budget=10)

        queue = asyncio.Queue()
        # only 1 new task with a budget of 15 should fit in the total budget
        for _ in range(2):
            queue.put_nowait(task1)

        new_tasks = policy.select_new_tasks(
            in_process_tasks=in_process_tasks,
            queue=queue,
            iterations_since_last_new_batch=0,
        )
        assert len(new_tasks) == 1
        assert new_tasks[0] == task1

        waiting_served_ratio = 2
        max_iterations_curr_batch = 2
        # test the waiting_served_ratio
        policy = QuotaBasedTaskSelectionPolicy(
            max_batch_total_tokens=45,  # 3 tasks can be scheduled
            max_batch_prefill_tokens=15,
            max_input_length=5,
            max_total_tokens=10,
            waiting_served_ratio=waiting_served_ratio,
            max_iterations_curr_batch=max_iterations_curr_batch,
        )

        expected_num_tasks_to_prefill = waiting_served_ratio * max_iterations_curr_batch

        num_in_process_tasks = max_iterations_curr_batch
        num_pending_tasks_to_prefill = policy._get_num_pending_tasks_to_prefill(
            num_in_process_tasks=num_in_process_tasks,
            iterations_since_last_new_batch=max_iterations_curr_batch,
        )
        assert num_pending_tasks_to_prefill == expected_num_tasks_to_prefill

        # test that new tasks aren't scheduled if there are tasks to be prefilled in
        # this case, num_tasks_to_prefill is 2 since max_iterations_curr_batch >
        new_tasks = policy.select_new_tasks(
            in_process_tasks=[task1, task1],
            queue=asyncio.Queue(),
            iterations_since_last_new_batch=max_iterations_curr_batch,  # num_tasks_to_prefill is 2
        )
        assert len(new_tasks) == 0

        # test that new tasks tasks aren't scheduled if there are less new tasks
        # to be scheduled than the number of tasks to be prefilled
        queue = asyncio.Queue()
        queue.put_nowait(task1)  # 1 pending task
        new_tasks = policy.select_new_tasks(
            in_process_tasks=[task1],  # 1 in process task
            queue=queue,
            iterations_since_last_new_batch=max_iterations_curr_batch,  # num_tasks_to_prefill is 2
        )
        assert len(new_tasks) == 0

        # test that when num_pending_tasks_to_prefill is 2 and there are 3 pending tasks
        # that are going to be scheduled, those 3 tasks are still scheduled.
        policy = QuotaBasedTaskSelectionPolicy(
            max_batch_total_tokens=60,  # 4 tasks can be scheduled
            max_batch_prefill_tokens=20,
            max_input_length=5,
            max_total_tokens=10,
            waiting_served_ratio=waiting_served_ratio,
            max_iterations_curr_batch=max_iterations_curr_batch,
        )

        queue = asyncio.Queue()
        for _ in range(3):
            queue.put_nowait(task1)
        new_tasks = policy.select_new_tasks(
            in_process_tasks=[task1],  # 1 in process task
            queue=queue,
            iterations_since_last_new_batch=max_iterations_curr_batch,  # num_tasks_to_prefill is 2
        )
        assert len(new_tasks) == 3
