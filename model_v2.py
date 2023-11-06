import torch 
import torch.nn as nn
import math
import torch.nn.functional as F

class InputEmbeddings(nn.Embedding):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__(vocab_size, d_model)
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        x = x.to(torch.int64)
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    def __init__(self, seq_len: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term) # 0::2 means 0, 2, 4, 6, ...
        pe[:, 1::2] = torch.cos(position * div_term) # 1::2 means 1, 3, 5, 7, ...

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.pe = pe

    def forward(self, x):
        self.pe[:, :x.shape[1], :].requires_grad = False
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)
    
class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x, mode):
        if mode=="train":
            return self.proj(x)
        else:
            return torch.log_softmax(self.proj(x), dim=-1)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """ one head of self-attention """
    def __init__(self, d_k: int, d_model: int, dropout: float):
        super().__init__()
        self.key = nn.Linear(d_model, d_k, bias=False)
        self.query = nn.Linear(d_model, d_k, bias=False)
        self.value = nn.Linear(d_model, d_k, bias=False)
        # self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        # B,T,C = x.shape
        k = self.key(key)   # (B,T,hs)
        q = self.query(query) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(mask == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(value) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

    
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, d_model, num_heads, head_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, d_model, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask):
        out = torch.cat([h(q, k, v, mask) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

    
class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

    
class EncoderBlock(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, d_model, num_heads, head_size, dropout):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        self.sa = MultiHeadAttention(d_model, num_heads, head_size, dropout)
        self.ffwd = FeedFoward(d_model, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, src_mask):
        x = self.ln1(x)
        x = x + self.sa(x, x, x, src_mask)
        x = x + self.ffwd(self.ln2(x))
        return x

    
class DecoderBlock(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, d_model, num_heads, head_size, dropout):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        self.sa = MultiHeadAttention(d_model, num_heads, head_size, dropout)
        self.ca = MultiHeadAttention(d_model, num_heads, head_size, dropout)
        self.ffwd = FeedFoward(d_model, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)

    def forward(self, x, src_mask, target_mask, encoder_output):
        x = self.ln1(x)
        x = x + self.sa(x, x, x, target_mask)
        x = self.ln2(x)
        x = x + self.ca(x, encoder_output, encoder_output, src_mask)
        x = x + self.ffwd(self.ln3(x))
        return x

    
class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, head_size, dropout, n_layer):
        super().__init__()
        self.layers = nn.Sequential(*[EncoderBlock(d_model, num_heads, head_size, dropout) for _ in range(n_layer)])
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.layer_norm(x)
    
    
class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, head_size, dropout, n_layer):
        super().__init__()
        self.layers = nn.Sequential(*[DecoderBlock( d_model, num_heads, head_size, dropout) for _ in range(n_layer)])
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, src_mask, target_mask, encoder_output):
        for layer in self.layers:
            x = layer(x, src_mask, target_mask, encoder_output)
        return self.layer_norm(x)
    

    
class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_emb: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, proj: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_emb = tgt_emb
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.proj = proj
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def encode(self, src, src_mask):
        return self.encoder(self.src_pos(self.src_embed(src)), src_mask)
    
    def decode(self, tgt, encoder_output, src_mask, tgt_mask):
        return self.decoder(self.tgt_pos(self.tgt_emb(tgt)), src_mask, tgt_mask, encoder_output)
    
    def project(self, x, mode="test"):
        return self.proj(x, mode)
    

def build_transformer(src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len, d_model, N, head, dropout):
    src_emb = InputEmbeddings(src_vocab_size, d_model)
    tgt_emb = InputEmbeddings(tgt_vocab_size, d_model)

    src_pos = PositionalEncoding(src_seq_len, d_model, dropout)
    tgt_pos = PositionalEncoding(tgt_seq_len, d_model, dropout)
    
    encoder = Encoder(d_model, head, d_model//head, dropout, N)
    decoder = Decoder(d_model, head, d_model//head, dropout, N)

    proj = ProjectionLayer(d_model, tgt_vocab_size)
    transformer = Transformer(encoder, decoder, src_emb, tgt_emb, src_pos, tgt_pos, proj)
    
    return transformer
