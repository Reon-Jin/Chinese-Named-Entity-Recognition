import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from vocab import Vocab
import utils
import math

class TransformerCRF(nn.Module):
    def __init__(self, sent_vocab, tag_vocab, dropout_rate=0.5, embed_size=768, hidden_size=768,
                 nhead=8, num_layers=12, dim_feedforward=3072):
        """ Initialize the model
        Args:
            sent_vocab (Vocab): vocabulary of words
            tag_vocab (Vocab): vocabulary of tags
            embed_size (int): embedding size
            hidden_size (int): hidden state size
            nhead (int): number of attention heads
            num_layers (int): number of transformer layers
            dim_feedforward (int): dimension of feedforward network
        """
        super(TransformerCRF, self).__init__()
        self.dropout_rate = dropout_rate
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.sent_vocab = sent_vocab
        self.tag_vocab = tag_vocab

        self.embedding = nn.Embedding(len(sent_vocab), embed_size)
        self.pos_encoding = PositionalEncoding(embed_size, dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)

        # 替换 LSTM 为 Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            batch_first=True  # 输入形状为 (batch, seq_len, features)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.hidden2emit_score = nn.Linear(embed_size, len(self.tag_vocab))
        self.transition = nn.Parameter(torch.randn(len(self.tag_vocab), len(self.tag_vocab)))  # shape: (K, K)

    def forward(self, sentences, tags, sen_lengths):
        """
        Args:
            sentences (tensor): sentences, shape (b, len). Lengths are in decreasing order, len is the length
                                of the longest sentence
            tags (tensor): corresponding tags, shape (b, len)
            sen_lengths (list): sentence lengths
        Returns:
            loss (tensor): loss on the batch, shape (b,)
        """
        mask = (sentences != self.sent_vocab[self.sent_vocab.PAD]).to(self.device)  # shape: (b, len)

        # 生成 Transformer 需要的 padding mask
        src_key_padding_mask = ~mask  # Transformer 中 True 表示需要被mask的位置

        # 嵌入层和位置编码
        embeddings = self.embedding(sentences)  # shape: (b, len, e)
        embeddings = self.pos_encoding(embeddings.transpose(0, 1)).transpose(0, 1)  # 添加位置编码

        emit_score = self.encode(embeddings, src_key_padding_mask)  # shape: (b, len, K)
        loss = self.cal_loss(tags, mask, emit_score)  # shape: (b,)
        return loss

    def encode(self, embeddings, src_key_padding_mask):
        """ Transformer Encoder
        Args:
            embeddings (tensor): sentences with word embeddings and positional encoding, shape (b, len, e)
            src_key_padding_mask (tensor): padding mask, shape (b, len)
        Returns:
            emit_score (tensor): emit score, shape (b, len, K)
        """
        # Transformer encoder
        hidden_states = self.encoder(embeddings, src_key_padding_mask=src_key_padding_mask)  # shape: (b, len, e)

        emit_score = self.hidden2emit_score(hidden_states)  # shape: (b, len, K)
        emit_score = self.dropout(emit_score)  # shape: (b, len, K)
        return emit_score

    def predict(self, sentences, sen_lengths):
        """
        Args:
            sentences (tensor): sentences, shape (b, len). Lengths are in decreasing order, len is the length
                                of the longest sentence
            sen_lengths (list): sentence lengths
        Returns:
            tags (list[list[str]]): predicted tags for the batch
        """
        batch_size = sentences.shape[0]
        mask = (sentences != self.sent_vocab[self.sent_vocab.PAD])  # shape: (b, len)
        src_key_padding_mask = ~mask

        # 嵌入层和位置编码
        embeddings = self.embedding(sentences)  # shape: (b, len, e)
        embeddings = self.pos_encoding(embeddings.transpose(0, 1)).transpose(0, 1)  # 添加位置编码

        emit_score = self.encode(embeddings, src_key_padding_mask)  # shape: (b, len, K)

        # 以下与原始代码相同
        tags = [[[i] for i in range(len(self.tag_vocab))]] * batch_size  # list, shape: (b, K, 1)
        d = torch.unsqueeze(emit_score[:, 0], dim=1)  # shape: (b, 1, K)
        for i in range(1, sen_lengths[0]):
            n_unfinished = mask[:, i].sum()
            d_uf = d[: n_unfinished]  # shape: (uf, 1, K)
            emit_and_transition = self.transition + emit_score[: n_unfinished, i].unsqueeze(dim=1)  # shape: (uf, K, K)
            new_d_uf = d_uf.transpose(1, 2) + emit_and_transition  # shape: (uf, K, K)
            d_uf, max_idx = torch.max(new_d_uf, dim=1)
            max_idx = max_idx.tolist()  # list, shape: (nf, K)
            tags[: n_unfinished] = [[tags[b][k] + [j] for j, k in enumerate(max_idx[b])] for b in range(n_unfinished)]
            d = torch.cat((torch.unsqueeze(d_uf, dim=1), d[n_unfinished:]), dim=0)  # shape: (b, 1, K)
        d = d.squeeze(dim=1)  # shape: (b, K)
        _, max_idx = torch.max(d, dim=1)  # shape: (b,)
        max_idx = max_idx.tolist()
        tags = [tags[b][k] for b, k in enumerate(max_idx)]
        return tags

    # cal_loss 方法保持不变
    def cal_loss(self, tags, mask, emit_score):
        """ Calculate CRF loss
        Args:
            tags (tensor): a batch of tags, shape (b, len)
            mask (tensor): mask for the tags, shape (b, len), values in PAD position is 0
            emit_score (tensor): emit matrix, shape (b, len, K)
        Returns:
            loss (tensor): loss of the batch, shape (b,)
        """
        batch_size, sent_len = tags.shape
        # calculate score for the tags
        score = torch.gather(emit_score, dim=2, index=tags.unsqueeze(dim=2)).squeeze(dim=2)  # shape: (b, len)
        score[:, 1:] += self.transition[tags[:, :-1], tags[:, 1:]]
        total_score = (score * mask.type(torch.float)).sum(dim=1)  # shape: (b,)
        # calculate the scaling factor
        d = torch.unsqueeze(emit_score[:, 0], dim=1)  # shape: (b, 1, K)
        for i in range(1, sent_len):
            n_unfinished = mask[:, i].sum()
            d_uf = d[: n_unfinished]  # shape: (uf, 1, K)
            emit_and_transition = emit_score[: n_unfinished, i].unsqueeze(dim=1) + self.transition  # shape: (uf, K, K)
            log_sum = d_uf.transpose(1, 2) + emit_and_transition  # shape: (uf, K, K)
            max_v = log_sum.max(dim=1)[0].unsqueeze(dim=1)  # shape: (uf, 1, K)
            log_sum = log_sum - max_v  # shape: (uf, K, K)
            d_uf = max_v + torch.logsumexp(log_sum, dim=1).unsqueeze(dim=1)  # shape: (uf, 1, K)
            d = torch.cat((d_uf, d[n_unfinished:]), dim=0)
        d = d.squeeze(dim=1)  # shape: (b, K)
        max_d = d.max(dim=-1)[0]  # shape: (b,)
        d = max_d + torch.logsumexp(d - max_d.unsqueeze(dim=1), dim=1)  # shape: (b,)
        llk = total_score - d  # shape: (b,)
        loss = -llk  # shape: (b,)
        return loss

    # save 和 load 方法需要调整以保存新的参数
    def save(self, filepath):
        params = {
            'sent_vocab': self.sent_vocab,
            'tag_vocab': self.tag_vocab,
            'args': dict(
                dropout_rate=self.dropout_rate,
                embed_size=self.embed_size,
                hidden_size=self.hidden_size,
                nhead=self.encoder.layers[0].self_attn.num_heads,
                num_layers=len(self.encoder.layers),
                dim_feedforward=self.encoder.layers[0].linear1.out_features
            ),
            'state_dict': self.state_dict()
        }
        torch.save(params, filepath)

    @staticmethod
    def load(filepath, device_to_load):
        params = torch.load(filepath, map_location=lambda storage, loc: storage)
        model = TransformerCRF(params['sent_vocab'], params['tag_vocab'], **params['args'])
        model.load_state_dict(params['state_dict'])
        model.to(device_to_load)
        return model

    @property
    def device(self):
        return self.embedding.weight.device


# 位置编码类
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=6000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def main():
    sent_vocab = Vocab.load('./vocab/sent_vocab.json')
    tag_vocab = Vocab.load('./vocab/tag_vocab.json')
    train_data, dev_data = utils.generate_train_dev_dataset('./data/NER-train-utf8.txt', sent_vocab, tag_vocab)
    device = torch.device('cpu')
    model = TransformerCRF(sent_vocab, tag_vocab)
    model.to(device)
    model.save('./model/transformer/model.pth')
    model = model.load('./model/transformer/model.pth', device)


if __name__ == '__main__':
    import math  # 需要导入math

    main()