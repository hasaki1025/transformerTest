from torch import nn

from TransformerImpl.model import attention


class AddNorm(nn.Module):
    def __init__(self, normalizer_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(kwargs)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(normalizer_shape)

    def forward(self, x, y):
        return self.norm(x + self.dropout(y))


class PositionWiseFFN(nn.Module):
    def __init__(self, input_size, output_size, num_hiddens, **kwargs):
        super(PositionWiseFFN, self).__init__(kwargs)
        self.fc1 = nn.Linear(input_size, num_hiddens)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(num_hiddens, output_size)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class EncoderBlock(nn.Module):
    def __init__(self, input_size, output_size, num_hiddens, num_heads, dropout, **kwargs):
        super(EncoderBlock, self).__init__(kwargs)
        self.attention = attention.MultiHeadAttention(num_heads, dropout, input_size, input_size, input_size,
                                                      num_hiddens)
        # TODO 为什么规范层的大小等于num_hiddens
        self.norm1 = AddNorm(normalizer_shape=num_hiddens, dropout=dropout)
        self.ffn = PositionWiseFFN(num_hiddens, output_size, num_hiddens)
        self.norm2 = AddNorm(normalizer_shape=output_size, dropout=dropout)

    def forward(self, x, mask=None):
        # TODO 这里是否需要mask掩盖
        y = self.attention(x, x, x, mask=mask)
        x = self.norm1(x, y)
        y = self.ffn(x)
        return self.norm2(x, y), mask


class DecoderBlock(nn.Module):
    def __init__(self, num_heads, dropout, input_size, num_hiddens, output_size, **kwargs):
        super(DecoderBlock, self).__init__(kwargs)
        self.attention1 = attention.MultiHeadAttention(num_heads, dropout, input_size, input_size, input_size,
                                                       num_hiddens)
        self.norm1 = AddNorm(normalizer_shape=num_hiddens, dropout=dropout)
        self.attention2 = attention.MultiHeadAttention(num_heads, dropout, num_hiddens, num_hiddens, num_hiddens,
                                                       num_hiddens)
        self.norm2 = AddNorm(normalizer_shape=num_hiddens, dropout=dropout)
        self.ffn = PositionWiseFFN(num_hiddens, output_size, num_hiddens)
        self.norm3 = AddNorm(normalizer_shape=output_size, dropout=dropout)

    def forward(self, x, encoder_outputs, dec_mask=None):
        x = self.norm1(x, self.attention1(x, x, x, mask=dec_mask))
        enc_out, enc_mask = encoder_outputs
        y = self.attention2(x, enc_out, enc_out, mask=enc_mask)
        x = self.norm2(x, y)
        return self.norm3(x, self.ffn(x))


class Encoder(nn.Module):
    def __init__(self, encoder_layers, d_model, num_heads, dropout, **kwargs):
        super(Encoder, self).__init__(kwargs)
        self.encoder_list = [EncoderBlock(d_model, d_model, d_model, num_heads, dropout) for i in range(encoder_layers)]
        self.embedding = nn.Embedding(num_embeddings=d_model, embedding_dim=d_model)


