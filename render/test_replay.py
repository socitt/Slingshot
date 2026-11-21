import re

import pytest

from env.navigation import FREE, RENDEZVOUS, TRIO, run_episode
from render.replay import View, draw, hud, render_episode

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def pd(obs):
    dx, dy, vx, vy = obs[0] * 2.0, obs[1] * 2.0, obs[2], obs[3]
    return [3 * dx - 3 * vx, 3 * dy - 3 * vy]


def test_view_maps_corners_and_centre():
    v = View(extent=2.0, width=20)
    assert v.height == 10
    assert v.cell(0.0, 0.0) == (5, 10)
    assert v.cell(-1.99, 1.99) == (0, 0)
    assert v.cell(1.99, -1.99) == (9, 19)
    assert v.cell(5.0, 0.0) is None
    with pytest.raises(ValueError):
        View(1.0, 5)


def test_draw_places_every_glyph():
    v = View(extent=2.0, width=40)
    frame = {"probe": [0.0, 0.0], "bodies": [1.0, 0.0, -1.0, 1.0], "thrust": (0.5, 0.0)}
    lines = draw(v, frame, [(0.5, -0.5)], (1.5, -1.5), 0.3, body_masses=(1.0, 0.01))
    text = "\n".join(lines)
    for glyph in "@*oX+.=":
        assert glyph in text, glyph
    assert len(lines) == v.height and all(len(l) == 40 for l in lines)
    # exhaust is drawn behind the probe (thrust points +x, so marker is left of @)
    row = next(l for l in lines if "@" in l)
    assert row.index("=") < row.index("@")


def test_color_only_adds_escapes():
    v = View(extent=2.0, width=30)
    frame = {"probe": [0.0, 0.0], "bodies": [], "thrust": (0.0, 0.0)}
    plain = draw(v, frame, [], (1.0, 1.0), 0.2)
    color = draw(v, frame, [], (1.0, 1.0), 0.2, color=True)
    assert [ANSI.sub("", l) for l in color] == plain
    assert color != plain


@pytest.mark.parametrize("task,policy", [(FREE, pd), (TRIO, lambda o: [0.0, 0.0]),
                                         (RENDEZVOUS, lambda o: [0.3, 0.0])])
def test_render_episode_frames_fit_the_width(task, policy):
    target = task.targets[0]
    result = run_episode(task, policy, target, record=True)
    masses = task.field()[0].masses if task.field else ()
    frames = list(render_episode(task, result, target, width=36, every=5, body_masses=masses))
    assert frames[-1][1] is True and not any(last for _, last in frames[:-1])
    for lines, _ in frames:
        assert all(len(l) <= 36 for l in lines), max(map(len, lines))
    final = "\n".join(frames[-1][0])
    assert ("REACHED" in final) == result.reached


def test_hud_reports_assists():
    class R:
        reached, crashed, out_of_bounds, score = True, False, False, 1.2
        work = [-0.9, 0.3, 0.1]
    lines = hud(R(), {"t": 1.0, "fuel_used": 0.25}, TRIO, 10, 10, 40)
    text = "\n".join(lines)
    assert "REACHED" in text and "gravity assist: +0.40" in text
    assert "fuel [" in text
