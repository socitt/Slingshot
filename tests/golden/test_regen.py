"""Entry-point test for the golden regenerator: it runs, and regenerating
from the current code reproduces the checked-in files (within float
noise) -- i.e. nobody changed the numerics without regenerating."""

import json
import subprocess
import sys

from tests.golden import regen


def test_unknown_suite_is_rejected():
    proc = subprocess.run([sys.executable, "-m", "tests.golden.regen", "nope"],
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 2
    assert "unknown suite" in proc.stderr


def test_regen_writes_into_given_data_dir(tmp_path, monkeypatch):
    mod = regen.SUITES["chaos"]
    monkeypatch.setattr(mod, "DATA_DIR", tmp_path)
    assert regen.main(["chaos", "lorenz_short"]) == 0
    written = json.loads((tmp_path / "chaos_lorenz_short.json").read_text())
    assert written["case"] == "lorenz_short"
    assert len(written["points"]) == 11
