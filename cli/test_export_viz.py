"""Entry-point test: run the real export_viz CLI as a subprocess."""

import json
import subprocess
import sys


def run(*args):
    return subprocess.run([sys.executable, "-m", "cli.export_viz", *args],
                          capture_output=True, text=True, timeout=240)


def test_selftest():
    proc = run("--selftest")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "export_viz: ok"


def test_quick_export_has_every_section(tmp_path):
    out = tmp_path / "data.json"
    proc = run("--out", str(out), "--quick")
    assert proc.returncode == 0, proc.stderr
    assert "wrote" in proc.stdout
    d = json.loads(out.read_text())
    assert {"agents", "divergence", "coast", "lyapunov", "lirk_log", "learning"} <= set(d)
    assert {"free", "trio", "assist"} <= set(d["agents"])
    assert d["learning"]["free"]["samples"][0]["gen"] == 0


def test_out_is_required():
    assert run().returncode == 2
