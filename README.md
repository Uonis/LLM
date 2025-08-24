# LLM
Educational PyTorch implementation demonstrating the first 3 layers of a GPT-style Transformer model. Includes tokenization, embeddings, sinusoidal positional encoding, multi-head self-attention. Designed for learning and visualizing how input text is processed through a Transformer architecture.



# Mini GPT-Style Multi-Head Attention (Educational)

This repository demonstrates the **Tokenization, Embedding, Positional Encoding, and Multi-Head Attention** part of a GPT-style Transformer model using PyTorch.  
It is designed for educational purposes to understand how input text is processed up to the multi-head attention mechanism.

## Features

- Tokenization using GPT2Tokenizer
- Token embedding layer
- Sinusoidal positional encoding
- Multi-Head Self-Attention
- Prints shapes and partial attention weights for inspection

## Code Overview

1. **Tokenizer**
   - Converts input text into token IDs
   - Pads sequences using the EOS token
   - Generates attention masks to indicate real tokens

2. **Embedding Layer**
   - Converts token IDs into dense vectors (`d_model`)

3. **Positional Encoding**
   - Adds positional information to embeddings so that the model knows the order of tokens

4. **Multi-Head Attention**
   - Each token attends to all other tokens
   - Multiple heads allow the model to focus on different aspects of the sentence
   - Causal mask ensures that tokens cannot attend to future tokens (important for GPT-style autoregressive models)

## How to Run

```bash
python mini_gpt_mult_head.py
