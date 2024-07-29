import spacy
import torch
from gensim import corpora
from gensim.models import Word2Vec
from torch.utils.data import Dataset, DataLoader, TensorDataset


def load_data_file(source_file, target_file):
    with open(source_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    source = [preprocess_sentence(line) for line in lines]
    with open(target_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    target = [preprocess_sentence(line) for line in lines]
    return TextDataSet(source, target)


def preprocess_sentence(sentence):
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # 使用空格替换不间断空格
    # 使用小写字母替换大写字母
    sentence = sentence.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # 在单词和标点符号之间插入空格
    out = [' ' + char if i > 0 and no_space(char, sentence[i - 1]) else char
           for i, char in enumerate(sentence)]
    return ''.join(out)


class TextDataSet(Dataset):

    def __init__(self, source, target):
        self.source = source
        self.target = target
        self.data = list(zip(source, target))

    def __len__(self):
        # 返回数据集的大小
        return len(self.data)

    def __getitem__(self, idx):
        # 根据索引返回数据
        src, trg = self.data[idx]
        return src, trg


class TokenizeDataset(Dataset):
    source_tokens: list[list[str]] = []
    target_tokens: list[list[str]] = []
    data: [(list[str], list[str])] = []

    def __init__(self, dataset):
        self.source_vocab = None
        self.target_vocab = None
        self.rawData = dataset
        self.init_token = '<bos>'
        self.end_token = '<eos>'
        self.pad_token = '<pad>'
        self.nlp_de = spacy.load('de_core_news_sm')
        self.nlp_en = spacy.load('en_core_web_sm')

    @staticmethod
    def is_valid_token(token):
        return not token.is_punct and not token.is_space

    def tokenize_en(self, text):
        source_doc = self.nlp_en(text)
        return [token.text for token in source_doc if TokenizeDataset.is_valid_token(token)]

    def tokenize_de(self, text):
        target_doc = self.nlp_de(text)
        return [token.text for token in target_doc if TokenizeDataset.is_valid_token(token)]

    def preprocess_text(self):
        for source, target in self.rawData:
            source_tokens = self.tokenize_de(source)
            target_tokens = self.tokenize_en(target)
            self.source_tokens.append(source_tokens)
            self.target_tokens.append(target_tokens)
            self.data.append((source_tokens, target_tokens))
        return self

    def build_vocab(self):
        spacial_words = [
            [self.init_token, self.end_token, self.pad_token],
        ]
        self.source_vocab = corpora.Dictionary(self.source_tokens + spacial_words)
        self.target_vocab = corpora.Dictionary(self.target_tokens + spacial_words)
        return self

    @staticmethod
    def id2vector(vocab_size, id):
        v = torch.zeros(vocab_size)
        v[id] = 1
        return v

    @staticmethod
    def array2onehot(array, vocab_size):
        result = torch.zeros(len(array), vocab_size)
        for i, id in enumerate(array):
            result[i, id] = 1
        return result

    def truncate_pad(self, vocab, sent_tensor, num_steps, is_target):
        init_vector, end_vector, pad_vector = (self.id2vector(len(vocab), vocab.token2id[self.init_token]).unsqueeze(0),
                                               self.id2vector(len(vocab), vocab.token2id[self.end_token]).unsqueeze(0),
                                               self.id2vector(len(vocab), vocab.token2id[self.pad_token]).unsqueeze(0))
        # TODO 是否需要添加bos标记
        if is_target:
            sent_tensor = torch.cat((init_vector, sent_tensor), dim=0)
        sent_tensor = torch.cat((sent_tensor, end_vector), dim=0)
        pad_len = 0
        while sent_tensor.shape[0] < num_steps:
            sent_tensor = torch.cat((sent_tensor, pad_vector), dim=0)
            pad_len += 1
        return sent_tensor[:num_steps], num_steps - pad_len

    def sentences2tensor(self, sentences, num_steps, vocab, is_target):
        def sentence2tensor(word_list):
            return self.array2onehot(
                vocab.doc2idx(word_list), len(vocab)
            )

        tensors = [sentence2tensor(sent) for sent in sentences]
        """将小批量语句转化为向量 ,返回值为三维张量和对应一维有效长度"""
        t = [
            self.truncate_pad(vocab, sent_tensor, num_steps, is_target) for sent_tensor in tensors
        ]

        return torch.stack(
            [h[0] for h in t], dim=0
        ).to(dtype=torch.int), torch.tensor([h[1] for h in t]).to(dtype=torch.int)

    def sentence2vector(self, sentences, vocab, num_steps, add_bos=False):
        init_toke_id, end_toke_id, pad_token_id = vocab.token2id['<bos>'], vocab.token2id['<eos>'], vocab.token2id[
            '<pad>']

        def sent2vector(word_list):
            result = [vocab.token2id[word] for word in word_list]
            if add_bos:
                result = [init_toke_id] + result
            result = result + [end_toke_id]
            padding_count = 0
            while len(result) < num_steps:
                result.append(pad_token_id)
                padding_count += 1
            return torch.tensor(result[:num_steps]).unsqueeze(0).to(dtype=torch.int), num_steps - padding_count

        res = [sent2vector(sent) for sent in sentences]
        vectors = [a[0] for a in res]
        valid_lens = [a[1] for a in res]
        return (torch.concat(list(vectors), dim=0),
                torch.tensor(list(valid_lens)).to(dtype=torch.int))

    def make_iter(self, num_steps, batch_size, is_train=True):
        #source_data, source_valid_lens = self.sentences2tensor(self.source_tokens, num_steps, self.source_vocab,False)
        #target_data, target_valid_lens = self.sentences2tensor(self.target_tokens, num_steps, self.target_vocab,True)
        source_data, source_valid_lens = self.sentence2vector(self.source_tokens, self.source_vocab, num_steps, False)
        target_data, target_valid_lens = self.sentence2vector(self.target_tokens, self.target_vocab, num_steps, True)
        dataset = TensorDataset(source_data, source_valid_lens, target_data, target_valid_lens)
        return DataLoader(dataset, batch_size=batch_size, shuffle=is_train)

# if __name__ == '__main__':
#     train_set = load_data_file('data/datasets/Multi30k/test.en', 'data/datasets/Multi30k/train.de')
#     tokenizeDataset = TokenizeDataset(train_set)
#     tokenizeDataset.preprocess_text().build_vocab()
#     dataloader = tokenizeDataset.make_iter(10, 1, True)
#     for batch in dataloader:
#         source_data, source_valid_lens, target_data, target_valid_lens = batch
#         print(batch)
