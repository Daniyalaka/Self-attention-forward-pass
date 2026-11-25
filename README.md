# Self-Attention Forward Pass

A from-scratch, NumPy-only implementation of single-head scaled dot-product
self-attention, demonstrated on the sentence "the cat sat on the mat".

## Formula
Attention(Q, K, V) = softmax(Q Kᵀ / √d_k) V

## Run
```bash
pip install numpy
python self_attention.py
```

## What it shows
- Q, K, V projections
- Scaled dot-product scores
- Optional causal mask (GPT-style) vs full attention (BERT-style)
- Softmax weights and the weighted sum of values
