"""Entry-point test: run the real train CLI as a subprocess."""

import json
import subprocess
import sys


def run(*args):
    return subprocess.run([sys.executable, "-m", "cli.train", *args],
                          capture_output=True, text=True, timeout=240)


def test_selftest():
    proc = run("--selftest")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "train: ok"


def test_short_training_run_writes_an_agent(tmp_path):
    out = tmp_path / "agent.json"
    proc = run("--generations", "3", "--pop", "6", "--seed", "3", "--out", str(out))
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.splitlines()
    assert [l.split()[1] for l in lines[:3]] == ["0", "1", "2"]
    assert all(len(l) <= 48 for l in lines if not l.startswith("saved ")), "phone width"
    assert "held-out targets" in proc.stdout
    saved = json.loads(out.read_text())
    assert saved["seed"] == 3 and saved["generations"] == 3
    assert len(saved["mlp"]["params"]) > 0


def test_curriculum_init_from_saved_agent(tmp_path):
    first = tmp_path / "a.json"
    assert run("--generations", "1", "--pop", "4", "--out", str(first), "--quiet").returncode == 0
    proc = run("--generations", "1", "--pop", "4", "--init", str(first), "--quiet")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.startswith("best ")


def test_bad_task_is_a_usage_error():
    proc = run("--task", "nope")
    assert proc.returncode == 2
