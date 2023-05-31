from aviary.common.backend import Backend, MockBackend, get_aviary_backend


def test_get_backend():
    backend = get_aviary_backend()
    assert backend
    assert isinstance(backend, Backend)


def test_get_mock():
    mock = MockBackend()
    assert mock

    assert mock.models()
    assert mock.completions("test", "test")
    assert mock.batch_completions(["test", "test"], "test")


def test_get_aviary():
    aviary = get_aviary_backend()
    assert aviary

    models = aviary.models()
    assert len(models) > 8
    assert "mosaicml/mpt-7b-instruct" in models

    completions = aviary.completions("test", "mosaicml/mpt-7b-instruct")
    assert completions
    assert "generated_text" in completions.keys()

    batch_completions = aviary.batch_completions(
        ["test", "test"], "mosaicml/mpt-7b-instruct"
    )
    assert all(
        "generated_text" in batch_completions[i] for i in range(len(batch_completions))
    )
