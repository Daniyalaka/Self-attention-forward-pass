"""
Self-Attention Forward Pass (NumPy, from scratch)
=================================================

Scenario: a tiny sentence  "the cat sat on the mat"  is turned into
embeddings, and we run ONE self-attention layer over it so every word
can "look at" the other words.

Steps of the forward pass
    1. Project input X into Queries (Q), Keys (K), Values (V)
    2. Scores   = Q @ K^T / sqrt(d_k)
    3. (optional) causal mask -> a word cannot see future words (GPT-style)
    4. Weights  = softmax(Scores)
    5. Output   = Weights @ V

Run:  python self_attention.py
"""

import numpy as np

np.random.seed(42)


def softmax(x, axis=-1):
    """Numerically stable softmax."""
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


class SelfAttention:
    """Single-head self-attention layer (forward pass only)."""

    def __init__(self, d_model, d_k):
        self.d_k = d_k
        # Learnable projection matrices (randomly initialised here)
        self.W_q = np.random.randn(d_model, d_k) * 0.1
        self.W_k = np.random.randn(d_model, d_k) * 0.1
        self.W_v = np.random.randn(d_model, d_k) * 0.1

    def forward(self, x, causal=False):
        """
        x: (T, d_model)  -> T tokens, each a d_model-dim vector
        returns: output (T, d_k), attention weights (T, T)
        """
        T = x.shape[0]

        # 1. Linear projections
        Q = x @ self.W_q          # (T, d_k)
        K = x @ self.W_k          # (T, d_k)
        V = x @ self.W_v          # (T, d_k)

        # 2. Scaled dot-product scores
        scores = Q @ K.T / np.sqrt(self.d_k)   # (T, T)

        # 3. Causal mask (hide future tokens)
        if causal:
            mask = np.tril(np.ones((T, T)))
            scores = np.where(mask == 0, -1e9, scores)

        # 4. Attention weights (each row sums to 1)
        weights = softmax(scores, axis=-1)     # (T, T)

        # 5. Weighted sum of values
        out = weights @ V                      # (T, d_k)
        return out, weights


def main():
    sentence = ["the", "cat", "sat", "on", "the", "mat"]
    T = len(sentence)
    d_model, d_k = 8, 4

    # Fake embeddings (in a real model these come from an embedding table)
    x = np.random.randn(T, d_model)

    attn = SelfAttention(d_model, d_k)

    for causal in (False, True):
        out, w = attn.forward(x, causal=causal)
        title = "CAUSAL (GPT-style)" if causal else "FULL (BERT-style)"
        print(f"\n=== {title} self-attention ===")
        print("Input shape :", x.shape)
        print("Output shape:", out.shape)
        print("Rows of weights sum to 1:", np.allclose(w.sum(axis=1), 1.0))

        print("\nAttention weights (row = word looking, column = word looked at):")
        header = "        " + " ".join(f"{t:>6}" for t in sentence)
        print(header)
        for i, row in enumerate(w):
            print(f"{sentence[i]:>6}  " + " ".join(f"{v:6.2f}" for v in row))


if __name__ == "__main__":
    main()
