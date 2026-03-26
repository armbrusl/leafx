from __future__ import annotations

import equinox as eqx


class Capture(eqx.Module):
    """Wraps an equinox module and marks its output for capture during leafx inference.

    Wrap any submodule with ``Capture`` to have leafx record the activation
    tensor produced by that module during a forward pass and display it in the
    graph view as a heatmap.

    Example::

        class MyModel(eqx.Module):
            encoder: leafx.Capture
            head: leafx.Capture

            def __init__(self, key):
                k1, k2 = jax.random.split(key)
                self.encoder = leafx.Capture(Encoder(key=k1), name="encoder")
                self.head    = leafx.Capture(nn.Linear(64, 10, key=k2), name="logits")

    :param module: The equinox module to wrap.
    :param name:   A human-readable label shown in the leafx graph.
    """

    module: eqx.Module
    name: str = ""

    def __call__(self, x, *args, **kwargs):
        return self.module(x, *args, **kwargs)
