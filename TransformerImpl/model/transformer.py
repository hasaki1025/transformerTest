import torch
from torch import nn

from TransformerImpl.model import attention


class AddNorm(nn.Module):
    def __init__(self, normalizer_shape, dropout):
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(normalizer_shape)

    def forward(self, x, y):
        return self.norm(x + self.dropout(y))


class PositionWiseFFN(nn.Module):
    def __init__(self, input_size, output_size, num_hiddens):
        super(PositionWiseFFN, self).__init__()
        self.fc1 = nn.Linear(input_size, num_hiddens)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(num_hiddens, output_size)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class EncoderBlock(nn.Module):
    def __init__(self, input_size, output_size, num_hiddens, num_heads, dropout):
        super(EncoderBlock, self).__init__()
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
    def __init__(self, num_heads, dropout, input_size, num_hiddens, output_size):
        super(DecoderBlock, self).__init__()
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


class PositionEncoder(nn.Module):
    def __init__(self, num_hiddens, dropout, device, max_len=1000):
        super(PositionEncoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        i = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1)
        j = torch.arange(0, num_hiddens, dtype=torch.float, device=device).unsqueeze(0)
        self.position = i / 10000 ** (2 * j / num_hiddens)
        self.position[:, 0::2] = torch.sin(self.position[:, 0::2])  # 这里利用了广播机制生成位置矩阵
        self.position[:, 1::2] = torch.cos(self.position[:, 1::2])

    def forward(self, x):
        x = x + self.position[x.shape[1], :].unsqueeze(0).to(device=x.device)
        return self.dropout(x)


def make_source_mask(x, valid_lens):
    """x为三维张量，valid_lens为一维张量, 需要注意的是掩码是对注意力矩阵有效，也就是对value生效，所以是掩盖列而不是行 , 最终掩码形状为(bathc_size, n, n)，其中n为时间步"""
    a = torch.arange(0, x.shape[1]).reshape(1, -1).unsqueeze(0).repeat((valid_lens.shape[0], 1, 1))
    return (a < valid_lens.reshape(-1, 1, 1)).repeat((1, x.shape[1], 1)).to(device=x.device)


def make_target_mask(y, valid_lens):
    mask_pad = make_source_mask(y, valid_lens)
    mask = mask_pad & torch.tril(torch.ones(y.shape[1], y.shape[1])).bool()
    return mask.to(y.device)


class Encoder(nn.Module):
    def __init__(self, encoder_layers, d_model, num_heads, dropout, device, vocab_size):
        super(Encoder, self).__init__()
        self.encoder_list = nn.ModuleList(
            [EncoderBlock(d_model, d_model, d_model, num_heads, dropout) for i in range(encoder_layers)])
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.PositionEncoder = PositionEncoder(d_model, dropout, device)

    def forward(self, x, valid_lens=None):
        x = self.embedding(x)
        x = self.PositionEncoder(x)
        mask = make_source_mask(x, valid_lens)
        for blk in self.encoder_list:
            x, _ = blk(x, mask)
        return x, mask


class Decoder(nn.Module):
    def __init__(self, decoder_layers, d_model, num_heads, dropout, device, vocab_size):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.PositionEncoder = PositionEncoder(d_model, dropout, device)
        self.decoder_list = nn.ModuleList(
            [DecoderBlock(num_heads, dropout, d_model, d_model, d_model) for i in range(decoder_layers)])

    def forward(self, x, encoder_outputs, valid_lens=None):
        x = self.embedding(x)
        x = self.PositionEncoder(x)
        dec_mask = make_target_mask(encoder_outputs, valid_lens)
        for blk in self.decoder_list:
            x = blk(x, encoder_outputs, dec_mask)
        return x


class Transformer(nn.Module):
    def __init__(self, num_heads, dropout, device, encoder_layers, d_model, decoder_layers, source_vocab_size,
                 target_vocab_size):
        super(Transformer, self).__init__()
        self.encoder = Encoder(encoder_layers, d_model, num_heads, dropout, device, source_vocab_size)
        self.decoder = Decoder(decoder_layers, d_model, num_heads, dropout, device, target_vocab_size)
        self.linear = nn.LazyLinear(out_features=target_vocab_size, bias=True, device=device)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, y, src_valid_lens=None, tar_valid_lens=None):
        if self.training:
            encoder_outputs = self.encoder(x, src_valid_lens)
            decoder_outputs = self.decoder(y, encoder_outputs, tar_valid_lens)
            return self.softmax(self.linear(decoder_outputs))
