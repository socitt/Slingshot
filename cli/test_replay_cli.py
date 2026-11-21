"""Entry-point test: run the real replay CLI as a subprocess."""

import subprocess
import sys

import pytest


def run(*args, env_extra=None):
    import os
    env = {**os.environ, "NO_COLOR": "1", **(env_extra or {})}
    return subprocess.run([sys.executable, "-m", "cli.replay", *args],
                          capture_output=True, text=True, timeout=120, env=env)


def test_selftest():
    proc = run("--selftest")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "replay: ok"


@pytest.mark.parametrize("task", ["free", "assist"])
def test_plain_replay_of_golden_agent(task):
    proc = run("--task", task, "--plain", "--width", "40")
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    assert "@" in out and "score" in out
    assert "\x1b[" not in out
    assert all(len(l) <= 40 for l in out.splitlines())


def test_animated_replay_uses_cursor_home():
    proc = run("--task", "free", "--fps", "1000", "--every", "20", "--width", "30")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.count("\x1b[H") >= 2


def test_bad_target_and_missing_agent():
    assert run("--task", "free", "--target", "99", "--plain").returncode == 2
    proc = run("--agent", "/nonexistent.json", "--plain")
    assert proc.returncode == 2 and "no agent" in proc.stderr
