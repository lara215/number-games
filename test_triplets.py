import pytest
from square import (
    solve_meshgrid,
    solve_chatGPT,
    solve_slow,
    solve_chatGPT_jonas,
)

SOLVERS = [solve_slow, solve_chatGPT, solve_chatGPT_jonas, solve_meshgrid]


@pytest.mark.parametrize("solver", SOLVERS)
def test_zero_limit_returns_empty(solver):
    """All solvers should return [] when limit is 0."""
    assert solver(0) == []