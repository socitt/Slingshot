"""Entry-point test: run the real chaos-demo as a subprocess."""

import subprocess
import sys

import pytest


def run(*args):
    return subprocess.run([sys.executable, "-m", "cli.chaos_demo", *args],
                          capture_output=True, text=True, timeout=120)


def test_selftest():
    proc = run("--selftest")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "chaos-demo: ok"


@pytest.mark.parametrize("cmd", ["bifurcation", "lorenz"])
def test_plots_fit_the_requested_width(cmd):
    proc = run(cmd, "--width", "36", "--height", "10", "--steps", "1500")
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.splitlines()
    assert len(lines) >= 12
    assert all(len(line) <= 36 for line in lines), max(map(len, lines))
    assert any(c in proc.stdout for c in "*#%@")


def test_lyapunov_table():
    proc = run("lyapunov", "--samples", "20000")
    assert proc.returncode == 0, proc.stderr
    assert "r=4" in proc.stdout and "+0.69" in proc.stdout
    assert "trace -13.667" in proc.stdout


def test_rejects_tiny_width():
    proc = run("lorenz", "--width", "5")
    assert proc.returncode == 2
    assert "width" in proc.stderr
