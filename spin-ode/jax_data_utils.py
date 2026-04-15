"""
jax_data_utils.py
-----------------
Jax data utilities
"""

from __future__ import annotations

from typing import Any, Generator, Optional, Protocol

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Dataset protocol
# ---------------------------------------------------------------------------


class Dataset(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> Any: ...  # returns a pytree


# ---------------------------------------------------------------------------
# Collate: list[pytree]  ->  batched pytree
# ---------------------------------------------------------------------------


def collate(items: list[Any]) -> Any:
    """
    Stack a list of same-structure pytrees into one batched pytree.

        [{"x": (4,), "y": ()}, ...]  ->  {"x": (B, 4), "y": (B,)}
        [(arr_a, arr_b), ...]        ->  (arr_a_batched, arr_b_batched)

    Each leaf across items is stacked with jnp.stack along a new axis 0.
    """
    return jax.tree.map(lambda *leaves: jnp.stack(leaves), *items)


# ---------------------------------------------------------------------------
# JAXDataLoader
# ---------------------------------------------------------------------------


class JAXDataLoader:
    """
    Parameters
    ----------
    dataset    : Dataset with __len__ and __getitem__ returning a pytree.
    batch_size : Samples per batch.
    shuffle    : Reshuffle indices each epoch using jax.random.
    drop_last  : Drop the final incomplete batch.
    key        : Base JAX PRNGKey for shuffling (default: PRNGKey(0)).
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
        key: Optional[jax.Array] = None,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self._key = key if key is not None else jax.random.PRNGKey(0)
        self._n = len(dataset)

    def __len__(self) -> int:
        if self.drop_last:
            return self._n // self.batch_size
        return -(-self._n // self.batch_size)  # ceil

    def __iter__(self) -> Generator[Any, None, None]:
        if self.shuffle:
            self._key, subkey = jax.random.split(self._key)
            indices = jax.random.permutation(subkey, self._n).tolist()
        else:
            indices = range(self._n)

        for start in range(0, self._n, self.batch_size):
            batch_idx = list(indices[start : start + self.batch_size])

            if self.drop_last and len(batch_idx) < self.batch_size:
                break

            items = [self.dataset[i] for i in batch_idx]  # list of pytrees
            yield collate(items)  # one batched pytree

    def forever(self) -> Generator[Any, None, None]:
        while True:
            yield from self


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    class ToyDataset:
        """Each item is a dict-pytree {"x": (4,), "label": ()}."""

        def __init__(self, n=20):
            key = jax.random.PRNGKey(0)
            self._x = jax.random.normal(key, (n, 4))
            self._y = jnp.arange(n)

        def __len__(self):
            return len(self._x)

        def __getitem__(self, i):
            return {"x": self._x[i], "label": self._y[i]}

    loader = JAXDataLoader(ToyDataset(20), batch_size=6, shuffle=True, drop_last=True)

    print(f"batches/epoch: {len(loader)}")
    for batch in loader:
        print(f"  x={batch['x'].shape}  label={batch['label'].shape}")
