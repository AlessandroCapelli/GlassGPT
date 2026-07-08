# GlassGPT

A complete GPT-style language model in a single, fully documented Python file: byte-level BPE tokenizer, multi-head causal self-attention, MLP blocks, a training loop, and text generation. Every layer is implemented explicitly with PyTorch tensors and autograd rather than through `nn.Transformer`, so each step of the architecture is visible and annotated.

## Quickstart

```bash
pip install -r requirements.txt
python glassgpt.py
```

With no `input.txt` present, GlassGPT trains on a small built-in sample (the
opening of Dante's _Inferno_) so the script runs out of the box, then prints
generated text.

To train on **your own data**, drop a UTF-8 text file named `input.txt` next to
`glassgpt.py` and run again. The trained model, its tokenizer and its config
are saved to `model.pt`; subsequent runs reload it instead of retraining.
Set `FORCE_RETRAIN = True` to retrain from scratch.

No GPU needed. CUDA is used automatically if available.

```
  TEXT
    |  BPE encode           (bytes -> merged tokens, outside the network)
    v
  idx (B, T)
    |  token embedding + position embedding      (WHAT + WHERE)
    v
  x (B, T, C)  <- the residual stream
    |  N x Block:
    |     x = x + MultiHeadAttention(LayerNorm(x))   (mix across positions)
    |     x = x + FeedForward(LayerNorm(x))          (per-position transform)
    v
  LayerNorm -> lm_head (C -> V)
    v
  logits (B, T, V)
    |  training:  cross-entropy vs targets -> backprop -> AdamW
    |  generate:  temperature -> top-k -> softmax -> sample
    v
  TEXT
```

## License

[MIT](LICENSE). Free to use, learn from, modify and share.
