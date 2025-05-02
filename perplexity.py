from __future__ import annotations

import flax
import jax.numpy as jnp
from clu.metrics import Metric
import optax


@flax.struct.dataclass
class Perplexity(Metric):
    """Computes perplexity from logits and integer labels.

    This assumes logits have shape [..., vocab_size] and labels have shape [...].
    """

    total_log_likelihood: jnp.ndarray
    total_tokens: jnp.ndarray

    @classmethod
    def empty(cls) -> Perplexity:
        return cls(
            total_log_likelihood=jnp.array(0.0, dtype=jnp.float32),
            total_tokens=jnp.array(0, dtype=jnp.int32)
        )

    @classmethod
    def from_model_output(
            cls, *, logits: jnp.ndarray, labels: jnp.ndarray, **kwargs
    ) -> Perplexity:
        if logits.ndim != labels.ndim + 1 or labels.dtype != jnp.int32:
            raise ValueError(
                f"Expected labels.dtype==jnp.int32 and logits.ndim={logits.ndim}=="
                f"labels.ndim+1={labels.ndim + 1}"
            )

        # Flatten for token-level cross entropy
        vocab_size = logits.shape[-1]
        logits_flat = logits.reshape(-1, vocab_size)
        labels_flat = labels.reshape(-1)

        # Negative log-likelihood per token
        nll = optax.softmax_cross_entropy_with_integer_labels(logits_flat, labels_flat)

        return cls(
            total_log_likelihood=jnp.sum(nll),
            total_tokens=labels_flat.size
        )

    def merge(self, other: Perplexity) -> Perplexity:
        return Perplexity(
            total_log_likelihood=self.total_log_likelihood + other.total_log_likelihood,
            total_tokens=self.total_tokens + other.total_tokens
        )

    def compute(self) -> jnp.ndarray:
        avg_nll = self.total_log_likelihood / self.total_tokens
        return jnp.exp(avg_nll)
