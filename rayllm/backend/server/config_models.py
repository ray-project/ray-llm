from typing import Optional
from pydantic import (
    BaseModel,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    field_validator,
)

# Adapted from ray.serve.config.AutoscalingConfig
# Port it here as the original AutoscalingConfig model is pydantic V1
class AutoscalingConfig(BaseModel):
    """Config for the Serve Autoscaler."""

    # Please keep these options in sync with those in
    # `src/ray/protobuf/serve.proto`.

    # Publicly exposed options
    min_replicas: NonNegativeInt = 1
    initial_replicas: Optional[NonNegativeInt] = None
    max_replicas: PositiveInt = 1

    # DEPRECATED: replaced by target_ongoing_requests
    target_num_ongoing_requests_per_replica: PositiveFloat = Field(
        default=1.0,
        description="[DEPRECATED] Please use `target_ongoing_requests` instead.",
    )
    # Will default to 1.0 in the future.
    target_ongoing_requests: Optional[PositiveFloat] = None

    # How often to scrape for metrics
    metrics_interval_s: PositiveFloat = 10.0
    # Time window to average over for metrics.
    look_back_period_s: PositiveFloat = 30.0

    # DEPRECATED
    smoothing_factor: PositiveFloat = 1.0
    # DEPRECATED: replaced by `downscaling_factor`
    upscale_smoothing_factor: Optional[PositiveFloat] = Field(
        default=None, description="[DEPRECATED] Please use `upscaling_factor` instead."
    )
    # DEPRECATED: replaced by `upscaling_factor`
    downscale_smoothing_factor: Optional[PositiveFloat] = Field(
        default=None,
        description="[DEPRECATED] Please use `downscaling_factor` instead.",
    )

    # Multiplicative "gain" factor to limit scaling decisions
    upscaling_factor: Optional[PositiveFloat] = None
    downscaling_factor: Optional[PositiveFloat] = None

    # How frequently to make autoscaling decisions
    # loop_period_s: float = CONTROL_LOOP_PERIOD_S
    # How long to wait before scaling down replicas
    downscale_delay_s: NonNegativeFloat = 600.0
    # How long to wait before scaling up replicas
    upscale_delay_s: NonNegativeFloat = 30.0

    @field_validator("max_replicas")
    def replicas_settings_valid(cls, max_replicas, values):
        min_replicas = values.data.get("min_replicas")
        initial_replicas = values.data.get("initial_replicas")
        if min_replicas is not None and max_replicas < min_replicas:
            raise ValueError(
                f"max_replicas ({max_replicas}) must be greater than "
                f"or equal to min_replicas ({min_replicas})!"
            )

        if initial_replicas is not None:
            if initial_replicas < min_replicas:
                raise ValueError(
                    f"min_replicas ({min_replicas}) must be less than "
                    f"or equal to initial_replicas ({initial_replicas})!"
                )
            elif initial_replicas > max_replicas:
                raise ValueError(
                    f"max_replicas ({max_replicas}) must be greater than "
                    f"or equal to initial_replicas ({initial_replicas})!"
                )

        return max_replicas

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
