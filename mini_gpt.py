# mini_gpt_mult_head.py
"""
Mini GPT-Style: Tokenization, Embedding, Positional Encoding, and Multi-Head Attention only.
"""

import math
import numpy as np
import torch
import torch.nn as nn
from transformers import GPT2TokenizerFast

# ---------- 1) Tokenizer ----------
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

text = "Ahmed went to the university"
tok = tokenizer(text, return_tensors="pt")
input_ids = tok["input_ids"]
attention_mask = tok["attention_mask"]

print("\n=== TOKENIZER ===")
print("Tokens:", tokenizer.convert_ids_to_tokens(input_ids[0].tolist()))
print("Input IDs shape:", tuple(input_ids.shape))
print("Vocabulary size:", tokenizer.vocab_size)

# ---------- Settings ----------
d_model   = 128
num_heads = 4
max_len   = 128

# ---------- 2) Embedding ----------
token_embed = nn.Embedding(tokenizer.vocab_size, d_model)
emb = token_embed(input_ids)

# ---------- 3) Positional Encoding ----------
def sinusoidal_positional_encoding(max_len, d_model):
    PE = np.zeros((max_len, d_model))
    for pos in range(max_len):
        for i in range(0, d_model, 2):
            angle = pos / (10000 ** (i / d_model))
            PE[pos, i] = math.sin(angle)
            if i + 1 < d_model:
                PE[pos, i + 1] = math.cos(angle)
    return torch.tensor(PE, dtype=torch.float32)

pos_enc_table = sinusoidal_positional_encoding(max_len, d_model)
seq_len = emb.size(1)
pe = pos_enc_table[:seq_len, :].unsqueeze(0)
x = emb + pe

print("\n=== EMBEDDING + POSITIONAL ENCODING ===")
print("x shape:", tuple(x.shape))
print("sample emb[0,0,:5]:", emb[0,0,:5])
print("sample pos [0,0,:5]:", pe[0,0,:5])

# ---------- 4) Multi-Head Attention ----------
class MultiHeadAttentionOnly(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=False)

    def causal_mask(self, seq_len):
        """Upper-triangular causal mask to prevent attending future tokens"""
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        return mask

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        x_t = x.transpose(0, 1)  # (seq_len, batch, d_model)
        attn_mask = self.causal_mask(seq_len).to(x.device)
        attn_out, attn_w = self.attn(x_t, x_t, x_t, attn_mask=attn_mask, need_weights=True, average_attn_weights=False)
        attn_out = attn_out.transpose(0,1)  # back to (batch, seq_len, d_model)
        return attn_out, attn_w

# ---------- Run Multi-Head Attention ----------
with torch.no_grad():
    mha = MultiHeadAttentionOnly(d_model, num_heads)
    out, attn_w = mha(x)
    print("\n=== MULTI-HEAD ATTENTION ===")
    print("Output shape:", tuple(out.shape))
    print("Attention weights shape:", tuple(attn_w.shape))
    print("First head attention (partial):\n", attn_w[0][:5,:5])
