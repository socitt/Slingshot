import pytest

from render.canvas import RAMP, density_plot, framed, terminal_width


def test_shape_is_exact():
    rows = density_plot([(0, 0), (1, 1)], width=17, height=5)
    assert len(rows) == 5
    assert all(len(r) == 17 for r in rows)


def test_y_axis_points_up():
    rows = density_plot([(0, 0), (1, 1)], width=4, height=4)
    assert rows[-1][0] != " "   # (0, 0) bottom-left
    assert rows[0][-1] != " "   # (1, 1) top-right
    assert rows[0][0] == " "


def test_density_ramp_orders_cells():
    pts = [(0.1, 0.5)] * 50 + [(0.9, 0.5)]
    row = density_plot(pts, width=2, height=1, xlim=(0, 1), ylim=(0, 1))[0]
    assert RAMP.index(row[0]) > RAMP.index(row[1]) > 0


def test_out_of_range_points_dropped_and_empty_ok():
    rows = density_plot([(5, 5)], width=3, height=2, xlim=(0, 1), ylim=(0, 1))
    assert rows == ["   ", "   "]
    assert density_plot([], 3, 2) == ["   ", "   "]


def test_degenerate_limits_do_not_divide_by_zero():
    rows = density_plot([(1, 1), (1, 1)], width=3, height=3)
    assert sum(c != " " for r in rows for c in r) == 1


def test_framed_width_and_title_truncation():
    out = framed(["abc", "def"], title="a very long title", footer="r")
    assert all(len(line) == 5 for line in out)
    assert out[1] == "|abc|"


def test_rejects_bad_size():
    with pytest.raises(ValueError):
        density_plot([(0, 0)], 0, 3)


def test_terminal_width_is_phone_friendly(monkeypatch):
    monkeypatch.setenv("COLUMNS", "200")
    assert terminal_width() == 60
    monkeypatch.setenv("COLUMNS", "10")
    assert terminal_width() == 20
