import torch
from torch import nn

import model
from TransformerImpl.dataloader.dataloader import load_data_file, TokenizeDataset
from TransformerImpl.model.transformer import Transformer





def train():
    def try_gpu(i=0):
        """如果存在，则返回gpu(i)，否则返回cpu()

        Defined in :numref:`sec_use_gpu`"""
        if torch.cuda.device_count() >= i + 1:
            return torch.device(f'cuda:{i}')
        return torch.device('cpu')

    num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10
    lr, num_epochs, device = 0.005, 200, try_gpu()
    ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4

    train_set = load_data_file('data/datasets/Multi30k/test.en', 'data/datasets/Multi30k/train.de')
    tokenizeDataset = TokenizeDataset(train_set)

    tokenizeDataset.preprocess_text().build_vocab()
    dataloader = tokenizeDataset.make_iter(num_steps, batch_size, True)

    source_vocab_size = len(tokenizeDataset.source_vocab)
    target_vocab_size = len(tokenizeDataset.target_vocab)

    loss = nn.CrossEntropyLoss()
    net = Transformer(num_heads, dropout, device, num_layers, num_hiddens, num_layers, source_vocab_size,target_vocab_size)
    optimizer = torch.optim.Adam(net.parameters(), lr)

    for i in range(num_epochs):
        net.train()
        epoch_loss = 0.0
        for x, x_valid_lens, y, y_valid_lens in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = net(x, y, x_valid_lens, y_valid_lens)
            l = loss(y_hat, y)
            epoch_loss += l
            l.backward()
            optimizer.step()

        print(f'epoch {i} loss: {epoch_loss} \n')


train()
