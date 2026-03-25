from __future__ import annotations

from pathlib import Path

import jax.random as jr
import jax.numpy as jnp
import equinox as eqx
import cloudpickle

import eqxview
from eqxview.model_loading import save_model_bundle


def make_mlp(
    in_size: int = 32,
    out_size: int = 4,
    width: int = 64,
    depth: int = 3,
    seed: int = 0,
):
    key = jr.PRNGKey(seed)

    model = eqx.nn.MLP(
        in_size=in_size,
        out_size=out_size,
        width_size=width,
        depth=depth,
        key=key,
    )

    cloudpickle.dumps(model)  # smoke test for pickling

    return model


def make_residual_mlp(
    in_size: int = 128,
    hidden: int = 256,
    n_blocks: int = 4,
    out_size: int = 10,
    seed: int = 0,
):
    return ComplexModel(
        in_size=in_size,
        hidden=hidden,
        n_blocks=n_blocks,
        out_size=out_size,
        key=jr.PRNGKey(seed),
    )


def make_conv_classifier(
    in_channels: int = 3,
    hidden: int = 16,
    out_size: int = 10,
    seed: int = 0,
):
    return ConvClassifier(
        in_channels=in_channels,
        hidden=hidden,
        out_size=out_size,
        key=jr.PRNGKey(seed),
    )


def make_token_attention(
    vocab: int = 256,
    d_model: int = 64,
    n_heads: int = 4,
    out_size: int = 8,
    seed: int = 0,
):
    return TokenAttentionModel(
        vocab=vocab,
        d_model=d_model,
        n_heads=n_heads,
        out_size=out_size,
        key=jr.PRNGKey(seed),
    )


def make_recurrent_cells(
    vocab: int = 128,
    d_model: int = 32,
    hidden: int = 48,
    out_size: int = 6,
    seed: int = 0,
):
    return RecurrentCellsModel(
        vocab=vocab,
        d_model=d_model,
        hidden=hidden,
        out_size=out_size,
        key=jr.PRNGKey(seed),
    )


class ResBlock(eqx.Module):
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    norm: eqx.nn.LayerNorm

    def __init__(self, features: int, *, key):
        k1, k2 = jr.split(key)
        self.linear1 = eqx.nn.Linear(features, features, key=k1)
        self.linear2 = eqx.nn.Linear(features, features, key=k2)
        self.norm = eqx.nn.LayerNorm((features,))

    def __call__(self, x):
        h = jnp.tanh(self.linear1(self.norm(x)))
        return x + self.linear2(h)


class ComplexModel(eqx.Module):
    encoder: eqx.nn.Linear
    blocks: list[ResBlock]
    head: eqx.nn.MLP

    def __init__(self, in_size=128, hidden=256, n_blocks=4, out_size=10, *, key):
        keys = jr.split(key, n_blocks + 2)
        self.encoder = eqx.nn.Linear(in_size, hidden, key=keys[0])
        self.blocks = [ResBlock(hidden, key=keys[i + 1]) for i in range(n_blocks)]
        self.head = eqx.nn.MLP(
            in_size=hidden,
            out_size=out_size,
            width_size=hidden // 2,
            depth=2,
            key=keys[-1],
        )

    def __call__(self, x):
        x = jnp.tanh(self.encoder(x))
        for block in self.blocks:
            x = block(x)
        return self.head(x)


class ConvClassifier(eqx.Module):
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    norm: eqx.nn.GroupNorm
    head: eqx.nn.Linear

    def __init__(self, in_channels=3, hidden=16, out_size=10, *, key):
        k1, k2, k3 = jr.split(key, 3)
        self.conv1 = eqx.nn.Conv2d(
            in_channels, hidden, kernel_size=3, padding=1, key=k1
        )
        self.conv2 = eqx.nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, key=k2)
        self.norm = eqx.nn.GroupNorm(groups=4, channels=hidden)
        self.head = eqx.nn.Linear(hidden, out_size, key=k3)

    def __call__(self, x):
        x = jnp.tanh(self.conv1(x))
        x = jnp.tanh(self.conv2(x))
        x = self.norm(x)
        x = jnp.mean(x, axis=(1, 2))
        return self.head(x)


class TokenAttentionModel(eqx.Module):
    embedding: eqxview.Capture
    attn: eqxview.Capture
    norm: eqx.nn.LayerNorm
    dropout: eqx.nn.Dropout
    head: eqxview.Capture

    def __init__(self, vocab=256, d_model=64, n_heads=4, out_size=8, *, key):
        k1, k2, k3 = jr.split(key, 3)
        self.embedding = eqxview.Capture(
            eqx.nn.Embedding(num_embeddings=vocab, embedding_size=d_model, key=k1),
            name="token_emb",
        )
        self.attn = eqxview.Capture(
            eqx.nn.MultiheadAttention(
                num_heads=n_heads,
                query_size=d_model,
                key_size=d_model,
                value_size=d_model,
                output_size=d_model,
                use_output_bias=True,
                dropout_p=0.0,
                inference=True,
                key=k2,
            ),
            name="attention",
        )
        self.norm = eqx.nn.LayerNorm((d_model,))
        self.dropout = eqx.nn.Dropout(p=0.1, inference=True)
        self.head = eqxview.Capture(
            eqx.nn.Linear(d_model, out_size, key=k3), name="logits"
        )

    def __call__(self, token_ids):
        x = self.embedding(token_ids)
        x = self.attn(x, x, x)
        x = jnp.stack([self.norm(tok) for tok in x], axis=0)
        x = self.dropout(x)
        return self.head(jnp.mean(x, axis=0))


class RecurrentCellsModel(eqx.Module):
    embedding: eqxview.Capture
    gru: eqxview.Capture
    lstm: eqxview.Capture
    head: eqxview.Capture
    hidden_size: int

    def __init__(self, vocab=128, d_model=32, hidden=48, out_size=6, *, key):
        k1, k2, k3, k4 = jr.split(key, 4)
        self.hidden_size = hidden
        self.embedding = eqxview.Capture(
            eqx.nn.Embedding(num_embeddings=vocab, embedding_size=d_model, key=k1),
            name="token_emb",
        )
        self.gru = eqxview.Capture(
            eqx.nn.GRUCell(d_model, hidden, key=k2), name="gru_hidden"
        )
        self.lstm = eqxview.Capture(
            eqx.nn.LSTMCell(hidden, hidden, key=k3), name="lstm_hidden"
        )
        self.head = eqxview.Capture(
            eqx.nn.Linear(hidden, out_size, key=k4), name="logits"
        )

    def __call__(self, token_ids):
        x = self.embedding(token_ids)
        h_gru = jnp.zeros((self.hidden_size,), dtype=x.dtype)
        h_lstm = (
            jnp.zeros((self.hidden_size,), dtype=x.dtype),
            jnp.zeros((self.hidden_size,), dtype=x.dtype),
        )
        for t in x:
            h_gru = self.gru(t, h_gru)
            h_lstm = self.lstm(h_gru, h_lstm)
        h_final, _ = h_lstm
        return self.head(h_final)


def build_model_zoo(seed: int = 42) -> dict[str, eqx.Module]:
    keys = jr.split(jr.PRNGKey(seed), 5)
    models: dict[str, eqx.Module] = {
        "mlp_small": make_mlp(in_size=32, out_size=4, width=64, depth=2, seed=seed),
        "residual_mlp": ComplexModel(
            in_size=128, hidden=256, n_blocks=4, out_size=10, key=keys[1]
        ),
        "conv_classifier": ConvClassifier(
            in_channels=3, hidden=16, out_size=10, key=keys[2]
        ),
        "token_attention": TokenAttentionModel(
            vocab=256, d_model=64, n_heads=4, out_size=8, key=keys[3]
        ),
        "recurrent_cells": RecurrentCellsModel(
            vocab=128, d_model=32, hidden=48, out_size=6, key=keys[4]
        ),
    }
    for name, model in models.items():
        cloudpickle.dumps(model)  # smoke test for pickling
    return models


def build_model_zoo_specs(seed: int = 42) -> dict[str, dict[str, object]]:
    return {
        "mlp_small": {
            "factory": "example_models:make_mlp",
            "factory_kwargs": {"in_size": 32, "out_size": 4, "width": 64, "depth": 2, "seed": seed},
        },
        "residual_mlp": {
            "factory": "example_models:make_residual_mlp",
            "factory_kwargs": {"in_size": 128, "hidden": 256, "n_blocks": 4, "out_size": 10, "seed": seed + 1},
        },
        "conv_classifier": {
            "factory": "example_models:make_conv_classifier",
            "factory_kwargs": {"in_channels": 3, "hidden": 16, "out_size": 10, "seed": seed + 2},
        },
        "token_attention": {
            "factory": "example_models:make_token_attention",
            "factory_kwargs": {"vocab": 256, "d_model": 64, "n_heads": 4, "out_size": 8, "seed": seed + 3},
        },
        "recurrent_cells": {
            "factory": "example_models:make_recurrent_cells",
            "factory_kwargs": {"vocab": 128, "d_model": 32, "hidden": 48, "out_size": 6, "seed": seed + 4},
        },
    }


def export_model_zoo(output_dir: Path | None = None, seed: int = 42) -> list[Path]:
    out = output_dir or Path(__file__).parent
    out.mkdir(parents=True, exist_ok=True)
    models = build_model_zoo(seed=seed)
    paths: list[Path] = []
    for name, model in models.items():
        path = out / f"{name}.pkl"
        with open(path, "wb") as f:
            cloudpickle.dump(model, f)
        paths.append(path)
    return paths


def export_model_bundle_zoo(output_dir: Path | None = None, seed: int = 42) -> list[Path]:
    out = output_dir or Path(__file__).parent
    out.mkdir(parents=True, exist_ok=True)
    models = build_model_zoo(seed=seed)
    specs = build_model_zoo_specs(seed=seed)
    paths: list[Path] = []
    for name, model in models.items():
        spec = specs[name]
        path = out / f"{name}.eqxbundle"
        save_model_bundle(
            model,
            path,
            factory_spec=spec["factory"],
            factory_kwargs=spec["factory_kwargs"],
        )
        paths.append(path)
    return paths


if __name__ == "__main__":
    saved_paths = export_model_bundle_zoo()
    for path in saved_paths:
        print(f"Saved {path.name} ({path.stat().st_size / 1024:.1f} KB)")
