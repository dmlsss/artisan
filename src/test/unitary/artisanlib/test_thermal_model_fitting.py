import numpy as np
import pytest

from artisanlib.thermal_model_fitting import _weighted_residual_objective


def test_weighted_residual_objective_matches_global_mse() -> None:
    r1 = np.array([1.0, 1.0], dtype=np.float64)
    r2 = np.array([3.0], dtype=np.float64)

    # Global MSE over all points: (1^2 + 1^2 + 3^2) / 3 = 11/3
    expected = 11.0 / 3.0
    assert _weighted_residual_objective([r1, r2]) == pytest.approx(expected)


def test_weighted_residual_objective_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        _weighted_residual_objective([])
