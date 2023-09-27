import time

import gevent
from locust import HttpUser, events
from locust.env import Environment
from locust.stats import stats_history, stats_printer


class RegressionTestBase:
    def __init__(self, user_class: HttpUser) -> None:
        """Initialize the regression test runner.

        Args:
            user_class: The class for the user to spawn in this test.

        """
        # setup Environment and Runner
        self._env = Environment(user_classes=[user_class], events=events)
        self._runner = self._env.create_local_runner()

        # execute init event handlers (only really needed if you have registered any)
        self._env.events.init.fire(environment=self._env, runner=self._runner)

        # start a greenlet that periodically outputs the current stats
        gevent.spawn(stats_printer(self._env.stats))

        # start a greenlet that save current stats to history
        gevent.spawn(stats_history, self._env.runner)

    def set_num_users(self, num_users: int, spawn_rate: int, wait: bool = False):
        """Set the number of users to currently"""
        self._runner.start(num_users, spawn_rate=spawn_rate)
        if wait:
            while True:
                time.sleep(0.1)
                if self._runner.user_count == num_users:
                    break

    def get_curr_num_users(self):
        return self._runner.user_count

    def get_stats(self):
        return self._env.stats.serialize_stats()

    def shutdown(self):
        self._runner.quit()
        self._runner.greenlet.join()
