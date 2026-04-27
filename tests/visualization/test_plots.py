import matplotlib
import numpy as np

matplotlib.use("Agg")

from forage_rl.visualization.plots import (
    _boundary_window_phase_sums,
    plot_boundary_window_recovery_comparison,
)


def test_boundary_window_phase_sums_ignore_center_step_and_nans():
    before_sum, after_sum = _boundary_window_phase_sums(
        np.array([1.0, 2.0, np.nan, 999.0, 4.0, np.nan, 8.0]),
        boundary_window_before=3,
        boundary_window_after=3,
    )

    assert before_sum == 3.0
    assert after_sum == 12.0


def test_plot_boundary_window_recovery_comparison_uses_before_after_bar_totals_without_error_bars():
    fig = plot_boundary_window_recovery_comparison(
        {
            "Policy A": [
                np.array([1.0, 1000.0, 9.0]),
                np.array([3.0, 2000.0, 11.0]),
            ]
        },
        boundary_window=1,
        save=False,
        show=False,
    )

    ax = fig.axes[0]
    bar_heights = sorted(float(bar.get_height()) for bar in ax.patches)

    np.testing.assert_allclose(bar_heights, np.array([2.0, 10.0]))
    assert len(ax.collections) == 0
