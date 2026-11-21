"""A tiny multilayer perceptron, from scratch, standard library only.

A network is fully described by its layer sizes plus one flat list of
floats (the *genome*): for each layer, the weight matrix row by row,
then the biases. Neuroevolution only ever sees that flat list.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

ACTIVATIONS = {
    "tanh": math.tanh,
    "relu": lambda v: v if v > 0.0 else 0.0,
    "linear": lambda v: v,
}


def param_count(sizes: list[int]) -> int:
    return sum(a * b + b for a, b in zip(sizes, sizes[1:]))


class MLP:
    """Feed-forward net. Hidden layers use ``hidden``, the output layer
    uses ``output`` (tanh by default, so actions land in [-1, 1])."""

    __slots__ = ("sizes", "hidden", "output", "_layers")

    def __init__(self, sizes: list[int], params: list[float],
                 hidden: str = "tanh", output: str = "tanh") -> None:
        if len(sizes) < 2 or any(s < 1 for s in sizes):
            raise ValueError(f"bad layer sizes {sizes}")
        if len(params) != param_count(sizes):
            raise ValueError(f"expected {param_count(sizes)} params for {sizes}, got {len(params)}")
        for name in (hidden, output):
            if name not in ACTIVATIONS:
                raise ValueError(f"unknown activation {name!r}")
        self.sizes, self.hidden, self.output = list(sizes), hidden, output
        # unpack once into (rows, biases) per layer: forward() is the hot loop
        layers, k = [], 0
        for a, b in zip(sizes, sizes[1:]):
            rows = [params[k + r * a: k + (r + 1) * a] for r in range(b)]
            k += a * b
            layers.append((rows, params[k:k + b]))
            k += b
        self._layers = layers

    @classmethod
    def random(cls, sizes: list[int], rng: random.Random, **kw) -> "MLP":
        return cls(sizes, init_params(sizes, rng), **kw)

    @property
    def params(self) -> list[float]:
        out: list[float] = []
        for rows, bias in self._layers:
            for row in rows:
                out.extend(row)
            out.extend(bias)
        return out

    def forward(self, x: list[float]) -> list[float]:
        if len(x) != self.sizes[0]:
            raise ValueError(f"expected {self.sizes[0]} inputs, got {len(x)}")
        hidden, last = ACTIVATIONS[self.hidden], len(self._layers) - 1
        act = x
        for i, (rows, bias) in enumerate(self._layers):
            f = ACTIVATIONS[self.output] if i == last else hidden
            act = [f(sum(w * v for w, v in zip(row, act)) + b) for row, b in zip(rows, bias)]
        return act

    __call__ = forward

    # --- persistence --------------------------------------------------
    def to_dict(self) -> dict:
        return {"sizes": self.sizes, "hidden": self.hidden,
                "output": self.output, "params": self.params}

    @classmethod
    def from_dict(cls, d: dict) -> "MLP":
        return cls(d["sizes"], d["params"], d.get("hidden", "tanh"), d.get("output", "tanh"))

    def save(self, path: str | Path, **extra) -> None:
        Path(path).write_text(json.dumps({**extra, "mlp": self.to_dict()}, indent=1) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "MLP":
        return cls.from_dict(json.loads(Path(path).read_text())["mlp"])


def init_params(sizes: list[int], rng: random.Random) -> list[float]:
    """Xavier/Glorot-style normal init for weights, zero biases."""
    out: list[float] = []
    for a, b in zip(sizes, sizes[1:]):
        std = math.sqrt(2.0 / (a + b))
        out.extend(rng.gauss(0.0, std) for _ in range(a * b))
        out.extend(0.0 for _ in range(b))
    return out
