#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import time
from pathlib import Path
from tqdm import tqdm

CONFIG = {
    'vocab_size': 10_000,
    'n_layers': 8,
    'n_heads': 8,
    'd_model': 512,
    'context_length': 256,
    'batch_size': 32,
    'learning_rate': 3e-4,
    'max_time_minutes': 50,
}

class SimpleTokenizer:
    def __init__(self, text=None):
        # Build character-level vocabulary from text if provided
        if text:
            chars = sorted(list(set(text)))
            self.vocab_size = len(chars)
            self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
            self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        
    def encode(self, text):
        # Convert text to list of token indices
        return [self.char_to_idx[c] for c in text if c in self.char_to_idx]
    
    def decode(self, indices):
        # Convert token indices back to text string
        return ''.join([self.idx_to_char[i] for i in indices if i in self.idx_to_char])
    
    def save(self, path):
        # Save tokenizer mappings to pickle file
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({'char_to_idx': self.char_to_idx, 'idx_to_char': self.idx_to_char}, f)
    
    def load(self, path):
        # Load tokenizer mappings from pickle file
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.char_to_idx = data['char_to_idx']
            self.idx_to_char = data['idx_to_char']
            self.vocab_size = len(self.char_to_idx)

class CausalAttention(nn.Module):
    # Multi-head self-attention with causal masking for autoregressive models
    def __init__(self, config):
        super().__init__()
        # Combined QKV projection for efficiency
        self.qkv = nn.Linear(config['d_model'], 3 * config['d_model'])
        self.proj = nn.Linear(config['d_model'], config['d_model'])
        self.n_heads = config['n_heads']
        self.d_model = config['d_model']
        
        # Causal mask prevents attending to future tokens
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config['context_length'], config['context_length']))
        )
        
    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.d_model, dim=2)
        
        # Reshape for multi-head attention
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        
        # Scaled dot-product attention with causal masking
        att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1) ** 0.5))
        att = att.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

class Block(nn.Module):
    # Transformer block: attention + MLP with residual connections
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config['d_model'])
        self.attn = CausalAttention(config)
        self.ln2 = nn.LayerNorm(config['d_model'])
        # MLP with 4x expansion factor
        self.mlp = nn.Sequential(
            nn.Linear(config['d_model'], 4 * config['d_model']),
            nn.GELU(),
            nn.Linear(4 * config['d_model'], config['d_model']),
        )
    
    def forward(self, x):
        # Pre-norm architecture with residual connections
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class MiniGPT(nn.Module):
    # GPT-style decoder-only transformer for language modeling
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embed = nn.Embedding(config['vocab_size'], config['d_model'])
        self.pos_embed = nn.Embedding(config['context_length'], config['d_model'])
        self.blocks = nn.ModuleList([Block(config) for _ in range(config['n_layers'])])
        self.ln_f = nn.LayerNorm(config['d_model'])
        self.head = nn.Linear(config['d_model'], config['vocab_size'], bias=False)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        # Initialize weights with small normal distribution
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        # Forward pass: embed tokens and positions, pass through blocks, project to vocab
        B, T = idx.size()
        tok_emb = self.token_embed(idx)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.pos_embed(pos)
        
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.head(x)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_tokens, temperature=0.8, top_k=40):
        # Autoregressive generation: sample tokens one at a time
        for _ in range(max_tokens):
            # Truncate to context length if needed
            idx_cond = idx[:, -self.config['context_length']:]
            logits, _ = self(idx_cond)
            # Take last position logits and apply temperature
            logits = logits[:, -1, :] / temperature
            
            # Top-k sampling: only consider top k tokens
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Sample from probability distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx

def train(data_path, config):
    # Train MiniGPT model on text data for specified time limit
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load text data, limit to 450MB for quick training
    print("Loading data...")
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()[:int(450 * 1024 * 1024)]
    
    print(f"Data size: {len(text):,} characters")
    
    # Build tokenizer from data
    tokenizer = SimpleTokenizer(text)
    config['vocab_size'] = tokenizer.vocab_size
    print(f"Vocabulary size: {config['vocab_size']}")
    
    # Tokenize entire dataset
    tokens = tokenizer.encode(text)
    print(f"Total tokens: {len(tokens):,}")
    
    # Initialize model
    model = MiniGPT(config).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {params:,}")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    
    # Training loop with time limit
    model.train()
    start_time = time.time()
    context_len = config['context_length']
    batch_size = config['batch_size']
    
    step = 0
    losses = []
    
    print(f"\nTraining for {config['max_time_minutes']} minutes...\n")
    
    try:
        while True:
            # Check if time limit reached
            elapsed = (time.time() - start_time) / 60
            if elapsed > config['max_time_minutes']:
                break
            
            # Sample random context windows from token sequence
            indices = torch.randint(0, len(tokens) - context_len - 1, (batch_size,))
            x = torch.stack([torch.tensor(tokens[i:i+context_len]) for i in indices]).to(device)
            y = torch.stack([torch.tensor(tokens[i+1:i+context_len+1]) for i in indices]).to(device)
            
            # Forward pass, backward pass, gradient clipping
            logits, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            losses.append(loss.item())
            step += 1
            
            # Log progress every 100 steps
            if step % 100 == 0:
                avg_loss = np.mean(losses[-100:])
                print(f"Step {step:5d} | Loss: {avg_loss:.4f} | Time: {elapsed:.1f}m")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted")
    
    print(f"\nTraining complete! Steps: {step:,} | Final loss: {losses[-1]:.4f}")
    
    # Save model checkpoint and tokenizer
    torch.save({
        'model': model.state_dict(),
        'config': config,
    }, 'model.pt')
    tokenizer.save('tokenizer.pkl')
    
    print("✓ Saved model.pt and tokenizer.pkl")
    
    return model, tokenizer

def generate_text(prompt, max_tokens=200):
    # Load trained model and generate text from prompt
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load checkpoint
    checkpoint = torch.load('model.pt', map_location=device)
    config = checkpoint['config']
    
    # Reconstruct model and load weights
    model = MiniGPT(config).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # Load tokenizer
    tokenizer = SimpleTokenizer()
    tokenizer.load('tokenizer.pkl')
    
    # Encode prompt and generate
    tokens = tokenizer.encode(prompt)
    idx = torch.tensor([tokens]).to(device)
    
    generated = model.generate(idx, max_tokens, temperature=0.8, top_k=40)
    text = tokenizer.decode(generated[0].tolist())
    
    return text

def main():
    # Command-line interface for training and generation
    parser = argparse.ArgumentParser(description='Minimal LLM Training')
    parser.add_argument('--data', type=str, help='Path to training data')
    parser.add_argument('--train', action='store_true', help='Train model')
    parser.add_argument('--generate', type=str, help='Generate from prompt')
    parser.add_argument('--max-tokens', type=int, default=200, help='Max tokens to generate')
    
    args = parser.parse_args()
    
    if args.train:
        if not args.data:
            print("Error: --data required for training")
            return
        train(args.data, CONFIG)
    
    elif args.generate:
        result = generate_text(args.generate, args.max_tokens)
        print("\n" + "="*60)
        print(result)
        print("="*60)
    
    else:
        print("Use --train to train or --generate to generate text")
        parser.print_help()

if __name__ == "__main__":
    main()
