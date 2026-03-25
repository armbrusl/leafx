from __future__ import annotations

from pathlib import Path

import jax.random as jr
import jax.numpy as jnp
import equinox as eqx
import cloudpickle


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
    embedding: eqx.nn.Embedding
    attn: eqx.nn.MultiheadAttention
    norm: eqx.nn.LayerNorm
    dropout: eqx.nn.Dropout
    head: eqx.nn.Linear

    def __init__(self, vocab=256, d_model=64, n_heads=4, out_size=8, *, key):
        k1, k2, k3 = jr.split(key, 3)
        self.embedding = eqx.nn.Embedding(
            num_embeddings=vocab, embedding_size=d_model, key=k1
        )
        self.attn = eqx.nn.MultiheadAttention(
            num_heads=n_heads,
            query_size=d_model,
            key_size=d_model,
            value_size=d_model,
            output_size=d_model,
            use_output_bias=True,
            dropout_p=0.0,
            inference=True,
            key=k2,
        )
        self.norm = eqx.nn.LayerNorm((d_model,))
        self.dropout = eqx.nn.Dropout(p=0.1, inference=True)
        self.head = eqx.nn.Linear(d_model, out_size, key=k3)

    def __call__(self, token_ids):
        x = self.embedding(token_ids)
        x = self.attn(x, x, x)
        x = jnp.stack([self.norm(tok) for tok in x], axis=0)
        x = self.dropout(x)
        return self.head(jnp.mean(x, axis=0))


class RecurrentCellsModel(eqx.Module):
    embedding: eqx.nn.Embedding
    gru: eqx.nn.GRUCell
    lstm: eqx.nn.LSTMCell
    head: eqx.nn.Linear
    hidden_size: int

    def __init__(self, vocab=128, d_model=32, hidden=48, out_size=6, *, key):
        k1, k2, k3, k4 = jr.split(key, 4)
        self.hidden_size = hidden
        self.embedding = eqx.nn.Embedding(
            num_embeddings=vocab, embedding_size=d_model, key=k1
        )
        self.gru = eqx.nn.GRUCell(d_model, hidden, key=k2)
        self.lstm = eqx.nn.LSTMCell(hidden, hidden, key=k3)
        self.head = eqx.nn.Linear(hidden, out_size, key=k4)

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


if __name__ == "__main__":
    saved_paths = export_model_zoo()
    for path in saved_paths:
        print(f"Saved {path.name} ({path.stat().st_size / 1024:.1f} KB)")
