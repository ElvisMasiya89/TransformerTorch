import math
import torch
import torch.nn as nn


class InputEmbeddings(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.word_embeddings = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.word_embeddings(x) * math.exp(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len, dropout):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (seq_len,d_model)
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimensions
        pe = pe.unsqueeze(0)  # (1, seq_len , d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.shape[1], :].requires_grad(False)  # (1, seq_len , d_model)
        return self.dropout(x)


class LayerNormalisation(nn.Module):
    def __init__(self, epsilon=10 ** -6):
        super().__init__()
        self.epsilon = epsilon

        # nn.Parameter makes the parameter learnable
        self.alpha = nn.Parameter(torch.ones(1))  # Multiplied
        self.bias = nn.Parameter(torch.zeros(1))  # Added

        def forward(self, x):
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True)
            return self.alpha * (x - mean) / (std + self.epsilon) + self.bias


class FeedForward(nn.Module):
    def __init__(self, d_model, dff, dropout):
        super().__init__()
        self.d_model = d_model
        self.dff = dff
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, dff)  # W1 + B1
        self.linear2 = nn.Linear(dff, d_model)  # W2 + B2

    def forward(self, x):
        # (Batch, se_len, d_model) -->(Batch, se_len, d_ff)-->(Batch, se_len, d_model)
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.attention_scores = None
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout

        self.d_k = d_model // self.num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.wo = nn.Linear(d_model, d_model)

        self.layer_norm1 = LayerNormalisation()
        self.layer_norm2 = LayerNormalisation()
        self.layer_norm3 = LayerNormalisation()

    @staticmethod
    def attention(q, k, v, mask, dropout):
        d_k = q.shape[-1]
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)  # (Batch, num_heads, Seq_Len,  Seq_Len)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        attn = torch.matmul(attention_scores, v)  # (Batch, num_heads, Seq_Len, d_k)
        return attn, attention_scores

    def forward(self, q, k, v, mask):
        q = self.wq(q)  # (Batch , Seq_Len, d_model) --> (Batch , Seq_Len, d_model)
        k = self.wk(k)  # (Batch , Seq_Len, d_model) --> (Batch , Seq_Len, d_model)
        v = self.wv(v)  # (Batch , Seq_Len, d_model) --> (Batch , Seq_Len, d_model)

        q = q.view(q.shape[0], q.shape[1], self.num_heads, self.d_k).transpose(1, 2)  # (Batch, num_heads, Seq_Len, d_k)
        k = k.view(k.shape[0], k.shape[1], self.num_heads, self.d_k).transpose(1, 2)  # (Batch, num_heads, Seq_Len, d_k)
        v = v.view(v.shape[0], v.shape[1], self.num_heads, self.d_k).transpose(1, 2)  # (Batch, num_heads, Seq_Len, d_k)

        x, self.attention_scores = MultiHeadAttention.attention(q, k, v, mask, self.dropout)
        # (Batch, num_heads, Seq_Len, d_k) --> # (Batch,  Seq_Len, num_heads, d_k) --> # (Batch, Seq_Len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], -1)
        x = self.wo(x)  # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_model)
        return x


class ResidualConnection(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = dropout
        self.layer_norm = LayerNormalisation()

    def forward(self, x, sub_layer):
        return x + self.dropout(sub_layer(self.layer_norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, dropout):
        super().__init__()
        self.dff = dff # Feed Forward Neural Network Output Size
        self.mha = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, dff, dropout)
        self.residual_mha = ResidualConnection(dropout)
        self.residual_ffn = ResidualConnection(dropout)

    def forward(self, x, mask):
        # Multi-Head Attention sub-layer
        attn_output = self.residual_mha(x, lambda x: self.mha(x, x, x, mask))

        # FeedForward sub-layer
        ffn_output = self.residual_ffn(attn_output, self.ffn)

        return ffn_output


class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, dropout):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout = dropout

        self.layer = nn.ModuleList([EncoderLayer(d_model, num_heads, dff, dropout) for _ in range(num_layers)])
        self.layer_norm = LayerNormalisation()

    def forward(self, x, mask=None):
        for i in range(self.num_layers):
            x = self.layer[i](x, mask)
        return self.layer_norm(x)


class DecoderLayer(nn.Module):

    def __init__(self, d_model, num_heads, dff, dropout):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff # Feed Forward Neural Network Output Size
        self.dropout = dropout

        self.mha = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_mha = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, dff, dropout)
        self.residual_mha = ResidualConnection(dropout)
        self.residual_cross_mha = ResidualConnection(dropout)
        self.residual_ffn = ResidualConnection(dropout)

    def forward(self, x, encoder_output, source_mask, target_mask):
        # Multi-Head Attention sub-layer
        attn_output = self.residual_mha(x, lambda x: self.mha(x, x, x, target_mask))

        # Cross-Attention sub-layer
        cross_attn_output = self.residual_cross_mha(attn_output,
                                                    lambda x: self.mha(x, encoder_output, encoder_output, source_mask))

        # FeedForward sub-layer
        ffn_output = self.residual_ffn(cross_attn_output, self.ffn)

        return ffn_output






