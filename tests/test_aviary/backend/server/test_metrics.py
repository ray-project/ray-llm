from aviary.backend.observability.fn_call_metrics import outer_product


def test_outer_product():
    assert outer_product([1, 2, 3], [1, 2, 3]) == [1, 2, 3, 4, 6, 9]
    assert outer_product([1, 5, 10], [1, 10, 100]) == [1, 5, 10, 50, 100, 500, 1000]
