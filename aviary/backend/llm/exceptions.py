class ValidationError(ValueError):
    pass


class PromptTooLongError(ValidationError):
    pass
