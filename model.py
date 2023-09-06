import torch 
import torch.nn as nn
import math

class LayerNormalization(nn.Module):
    def __init__(self, eps=1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
        return x


class InputEmbeddings(nn.Embedding):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
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

    def forward(self, x):
        x = x + self.pe[:, :x.shape[1], :].requires_grad(False)
        return self.dropout(x)
    
class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.layer_norm(x)))

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model, h, dropout) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h

        assert d_model % h == 0

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
    
    @staticmethod
    def attention(query, key, value, mask=None, dropout=None):
        d_k = query.shape[-1]
        attention_scores = query @ key.transpose(-2, -1) / math.sqrt(d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        attention_probs = torch.softmax(attention_scores, dim=-1)
        if dropout is not None:
            attention_probs = dropout(attention_probs)
        
        return attention_probs @ value, attention_probs

    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_probs = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.d_model)
        return self.w_o(x)
    

class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        return self.residual_connections[1](x, self.feed_forward_block)
    

class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.layer_norm = LayerNormalization()

    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.layer_norm(x)
    

class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, src_mask, tgt_mask, encoder_output):
        x = self.residual_connections[0](x, self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block(x))
        return x
    
class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.layer_norm = LayerNormalization()

    def forward(self, x, src_mask, tgt_mask, encoder_output):
        for layer in self.layers:
            x = layer(x, src_mask, tgt_mask, encoder_output)
        return self.layer_norm(x)
    
class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim=-1)
    

class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_emb: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, proj: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_emb = tgt_emb
        self.pos = pos
        self.proj = proj

    def encode(self, src, src_mask):
        return self.encoder(self.pos(self.src_embed(src)), src_mask)
    
    def decode(self, tgt, encoder_output, src_mask, tgt_mask):
        return self.decoder(self.pos(self.tgt_emb(tgt)), src_mask, tgt_mask, encoder_output)
    
    def project(self, x):
        return self.proj(x)
    

def build_transformer(src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len, d_model, N, head, dropout, d_ff):
    src_emb = InputEmbeddings(src_vocab_size, d_model)
    tgt_emb = InputEmbeddings(tgt_vocab_size, d_model)

    src_pos = PositionalEncoding(src_seq_len, d_model, dropout)
    tgt_pos = PositionalEncoding(tgt_seq_len, d_model, dropout)

    encoder_blocks = []
    for i in range(N):
        attn = MultiHeadAttentionBlock(d_model, head, dropout)
        ff = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_blocks.append(EncoderBlock(attn, ff, dropout))
    encoder = Encoder(nn.ModuleList(encoder_blocks))

    decoder_blocks = []
    for i in range(N):
        self_attn = MultiHeadAttentionBlock(d_model, head, dropout)
        cross_attn = MultiHeadAttentionBlock(d_model, head, dropout)
        ff = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_blocks.append(DecoderBlock(self_attn, cross_attn, ff, dropout))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    proj = ProjectionLayer(d_model, tgt_vocab_size)
    transformer = Transformer(encoder, decoder, src_emb, tgt_emb, src_pos, tgt_pos, proj)
    
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            
    return transformer






    
