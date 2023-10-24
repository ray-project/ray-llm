# This is a lightweight fork of Opentelemetry's Threading instrmentor which
# since it hasn't been merged yet, we're pulling in directly. Pulled from
# https://github.com/open-telemetry/opentelemetry-python-contrib/pull/1582 on
# 2023-09-12 by thomas@.

# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=empty-docstring,no-value-for-parameter,no-member,no-name-in-module

import threading  # pylint: disable=import-self
from typing import Collection

from opentelemetry import context, trace
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.trace import (
    get_current_span,
    get_tracer,
    get_tracer_provider,
)


class _InstrumentedThread(threading.Thread):
    _tracer: trace.Tracer
    _parent_span: trace.Span

    def start(self):
        self._parent_span = get_current_span()
        super().start()

    def run(self):
        parent_span = self._parent_span or get_current_span()
        ctx = trace.set_span_in_context(parent_span)
        context.attach(ctx)
        super().run()


class ThreadingInstrumentor(BaseInstrumentor):  # pylint: disable=empty-docstring
    original_threadcls = threading.Thread

    def instrumentation_dependencies(self) -> Collection[str]:
        return ()

    def _instrument(self, *args, **kwargs):
        tracer_provider = kwargs.get("tracer_provider", None) or get_tracer_provider()

        tracer = get_tracer(__name__, "0.0.1", tracer_provider)
        threading.Thread = _InstrumentedThread
        _InstrumentedThread._tracer = tracer

    def _uninstrument(self, **kwargs):
        threading.Thread = self.original_threadcls
