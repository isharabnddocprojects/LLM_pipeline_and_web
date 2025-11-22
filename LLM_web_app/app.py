import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from pathlib import Path
from flask import Flask, request, jsonify, render_template

class SimpleTokenizer(object):
    # Character-level tokenizer implementation for text encoding/decoding
    # Maps characters to integer indices and vice versa for model input/output
    def __init__(self, text=None):
        # Build vocabulary from unique characters in text if provided
        # Otherwise initialize empty mappings for loading from file
        if text:
            chars = sorted(list(set(text)))
            self.vocab_size = len(chars)
            self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
            self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        else:
            self.char_to_idx = {}
            self.idx_to_char = {}
            self.vocab_size = 0
            
    def encode(self, text):
        # Convert text string to list of token indices, using 0 for unknown chars
        return [self.char_to_idx.get(c, 0) for c in text if c in self.char_to_idx]
    
    def decode(self, indices):
        # Convert token indices back to text string, using '?' for unknown indices
        return ''.join([self.idx_to_char.get(i, '?') for i in indices if i in self.idx_to_char])
    
    def save(self, path):
        # Persist tokenizer mappings to disk using pickle for model deployment
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({'char_to_idx': self.char_to_idx, 'idx_to_char': self.idx_to_char}, f)
    
    def load(self, path):
        # Restore tokenizer mappings from saved pickle file
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.char_to_idx = data['char_to_idx']
            self.idx_to_char = data['idx_to_char']
            self.vocab_size = len(self.char_to_idx)

class CausalAttention(nn.Module):
    # Multi-head self-attention with causal masking for autoregressive generation
    def __init__(self, config):
        super().__init__()
        # Combined QKV projection for efficiency (3x d_model output split into q, k, v)
        self.qkv = nn.Linear(config['d_model'], 3 * config['d_model'])
        # Output projection after attention computation
        self.proj = nn.Linear(config['d_model'], config['d_model'])
        self.n_heads = config['n_heads']
        self.d_model = config['d_model']
        
        # Causal mask: lower triangular matrix prevents attending to future tokens
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config['context_length'], config['context_length']))
        )
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, d_model)
        B, T, C = x.size()
        # Compute Q, K, V from input in single pass
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.d_model, dim=2)
        
        # Reshape for multi-head attention: split d_model into n_heads
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        
        # Scaled dot-product attention: QK^T / sqrt(d_k)
        att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1) ** 0.5))
        # Apply causal mask: set future positions to -inf before softmax
        att = att.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        
        # Weighted sum of values using attention scores
        y = att @ v
        # Concatenate heads back to original d_model dimension
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

class Block(nn.Module):
    # Transformer block: attention + MLP with residual connections and layer norm
    def __init__(self, config):
        super().__init__()
        # Pre-norm architecture: normalize before attention/MLP
        self.ln1 = nn.LayerNorm(config['d_model'])
        self.attn = CausalAttention(config)
        self.ln2 = nn.LayerNorm(config['d_model'])
        # MLP with 4x expansion: standard transformer feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(config['d_model'], 4 * config['d_model']),
            nn.GELU(),
            nn.Linear(4 * config['d_model'], config['d_model']),
        )
    
    def forward(self, x):
        # Residual connection around attention: x = x + attention(norm(x))
        x = x + self.attn(self.ln1(x))
        # Residual connection around MLP: x = x + mlp(norm(x))
        x = x + self.mlp(self.ln2(x))
        return x

class MiniGPT(nn.Module):
    # GPT-style decoder-only transformer for autoregressive language modeling
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Token embeddings: map token indices to d_model dimensional vectors
        self.token_embed = nn.Embedding(config['vocab_size'], config['d_model'])
        # Position embeddings: learnable positional encoding for sequence order
        self.pos_embed = nn.Embedding(config['context_length'], config['d_model'])
        # Stack of transformer blocks
        self.blocks = nn.ModuleList([Block(config) for _ in range(config['n_layers'])])
        # Final layer norm before output projection
        self.ln_f = nn.LayerNorm(config['d_model'])
        # Output head: projects to vocabulary size for next-token prediction
        self.head = nn.Linear(config['d_model'], config['vocab_size'], bias=False)
        
        # Initialize weights using standard transformer initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        # Weight initialization: normal distribution for linear layers and embeddings
        # Small std (0.02) prevents large initial activations
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        # idx: token indices of shape (batch_size, sequence_length)
        B, T = idx.size()
        # Embed tokens and positions, then combine
        tok_emb = self.token_embed(idx)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.pos_embed(pos)
        
        # Add positional embeddings to token embeddings
        x = tok_emb + pos_emb
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm and projection to vocabulary
        x = self.ln_f(x)
        logits = self.head(x)
        
        # Compute cross-entropy loss if targets provided (training mode)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_tokens, temperature=0.8, top_k=40):
        # Autoregressive text generation: sample tokens one at a time
        for _ in range(max_tokens):
            # Truncate to context length if sequence exceeds it
            idx_cond = idx[:, -self.config['context_length']:]
            logits, _ = self(idx_cond)
            # Take logits for last position only (next token prediction)
            logits = logits[:, -1, :] / temperature
            
            # Top-k sampling: only consider top k most likely tokens
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Convert logits to probabilities and sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append sampled token to sequence
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx

app = Flask(__name__)
device = 'cpu'

# Global model instances loaded at startup for inference
MODEL = None
TOKENIZER = None
CONFIG = None

MODEL_OLD = None
TOKENIZER_OLD = None


def load_model():
    # Load both new (augmented) and old (original) models for side-by-side comparison
    global MODEL, TOKENIZER, MODEL_OLD, TOKENIZER_OLD, CONFIG
    
    model_path = Path('./models/model.pt')
    tokenizer_path = Path('./models/tokenizer.pkl')
    model_old_path = Path('./models_old/model.pt')
    tokenizer_old_path = Path('./models_old/tokenizer.pkl')
    
    print(f"Loading NEW and OLD models on device: {device}...")
    
    try:
        # Load new model checkpoint and configuration
        checkpoint = torch.load(model_path, map_location=device)
        CONFIG = checkpoint['config'] 
        
        # Initialize and load tokenizer for new model
        tokenizer = SimpleTokenizer()
        tokenizer.load(tokenizer_path)
        TOKENIZER = tokenizer

        # Create model instance and load weights, set to eval mode
        MODEL = MiniGPT(CONFIG).to(device)
        MODEL.load_state_dict(checkpoint['model'])
        MODEL.eval()
        print(f"NEW Model (Augmented) loaded successfully. Vocab Size: {CONFIG['vocab_size']}")

        # Load old model checkpoint and configuration
        checkpoint_old = torch.load(model_old_path, map_location=device)
        CONFIG_OLD = checkpoint_old['config'] 
        
        # Initialize and load tokenizer for old model
        tokenizer_old = SimpleTokenizer()
        tokenizer_old.load(tokenizer_old_path)
        TOKENIZER_OLD = tokenizer_old

        # Create old model instance and load weights, set to eval mode
        MODEL_OLD = MiniGPT(CONFIG_OLD).to(device) 
        MODEL_OLD.load_state_dict(checkpoint_old['model'])
        MODEL_OLD.eval()
        print(f"OLD Model (Original) loaded successfully. Vocab Size: {CONFIG_OLD['vocab_size']}")
        
    except Exception as e:
        print(f"ERROR: Could not load model files. Check if ./models/ and ./models_old/ exist and contain model.pt/tokenizer.pkl. Error: {e}")
        exit()

with app.app_context():
    load_model()

@app.route('/')
def index():
    # Serve main web interface for model comparison
    return render_template('index.html', device=device)

@app.route('/health')
def health():
    # Health check endpoint for load balancer monitoring
    if MODEL and MODEL_OLD:
        return jsonify({"status": "healthy", "models_loaded": True}), 200
    else:
        return jsonify({"status": "unhealthy", "models_loaded": False}), 503

@app.route('/generate', methods=['POST'])
@torch.no_grad()
def generate_text_endpoint():
    # API endpoint: generate text from both models for comparison
    data = request.json
    prompt = data.get('prompt', '')
    max_tokens = data.get('max_tokens', 100)
    
    print(f"Request received. Prompt: '{prompt[:40]}...'")
    
    if not MODEL or not MODEL_OLD:
        return jsonify({"error": "One or both models failed to load."}), 500

    try:
        # Generate with new model: encode prompt, generate, decode
        tokens = TOKENIZER.encode(prompt)
        idx = torch.tensor([tokens], dtype=torch.long, device=device)
        
        print(f"Generating NEW model response (Max {max_tokens} tokens)...")
        generated = MODEL.generate(idx, max_tokens, temperature=0.5, top_k=40)
        generated_text = TOKENIZER.decode(generated[0].tolist())

        # Generate with old model: encode prompt, generate, decode
        tokens_old = TOKENIZER_OLD.encode(prompt)
        idx_old = torch.tensor([tokens_old], dtype=torch.long, device=device)
        
        print(f"Generating OLD model response (Max {max_tokens} tokens)...")
        generated_old = MODEL_OLD.generate(idx_old, max_tokens, temperature=0.5, top_k=40)
        generated_text_old = TOKENIZER_OLD.decode(generated_old[0].tolist())
        
        print("--- Generation Complete ---")
        # Return both responses for side-by-side comparison
        return jsonify({
            "prompt": prompt, 
            "response": generated_text,
            "response_old": generated_text_old,
        })
        
    except Exception as e:
        print(f"ERROR DURING GENERATION: {e}")
        return jsonify({"error": f"Generation failed: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
