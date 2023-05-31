def has_ray():
    try:
        import ray  # noqa: F401

        return True
    except ImportError:
        return False


def has_backend():
    try:
        import aviary.backend  # noqa: F401

        return True
    except ImportError:
        return False


def assert_has_ray():
    assert has_ray(), (
        "This command requires ray to be installed. "
        "Please install ray with `pip install 'ray[default]'`"
    )


def assert_has_backend():
    assert has_backend(), (
        "This command requires aviary backend to be installed. "
        "Please install backend dependencies with `pip install aviary[backend]`. "
    )
