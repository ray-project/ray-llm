from aviary.backend.observability.tracing import setup_tracing

setup_tracing()

from aviary.conf import secrets  # noqa
from aviary.sdk import *  # noqa: E402
