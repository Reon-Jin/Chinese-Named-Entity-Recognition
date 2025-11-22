import torch
import torch.nn as nn
from vocab import Vocab
import utils


class CRF(nn.Module):
    def __init__(self, sent_vocab, tag_vocab, dropout_rate=0.2, embed_size=256):
        """ Initialize the model
        Args:
            sent_vocab (Vocab): vocabulary of words
            tag_vocab (Vocab): vocabulary of tags
            embed_size (int): embedding size
        """
        super(CRF, self).__init__()
        self.dropout_rate = dropout_rate
        self.embed_size = embed_size
        self.sent_vocab = sent_vocab
        self.tag_vocab = tag_vocab

        # 词嵌入层
        self.embedding = nn.Embedding(len(sent_vocab), embed_size)
        self.dropout = nn.Dropout(dropout_rate)

        # 直接通过线性层从词嵌入得到发射分数
        self.embed2emit_score = nn.Linear(embed_size, len(self.tag_vocab))

        # 转移矩阵参数
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
        emit_score = self.encode(sentences)  # shape: (b, len, K)
        loss = self.cal_loss(tags, mask, emit_score)  # shape: (b,)
        return loss

    def encode(self, sentences):
        """ 直接从词嵌入得到发射分数
        Args:
            sentences (tensor): sentences, shape (b, len)
        Returns:
            emit_score (tensor): emit score, shape (b, len, K)
        """
        # 获取词嵌入
        embeddings = self.embedding(sentences)  # shape: (b, len, e)
        embeddings = self.dropout(embeddings)

        # 通过线性层得到发射分数
        emit_score = self.embed2emit_score(embeddings)  # shape: (b, len, K)
        return emit_score

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
        emit_score = self.encode(sentences)  # shape: (b, len, K)
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

    def save(self, filepath):
        params = {
            'sent_vocab': self.sent_vocab,
            'tag_vocab': self.tag_vocab,
            'args': dict(dropout_rate=self.dropout_rate, embed_size=self.embed_size),
            'state_dict': self.state_dict()
        }
        torch.save(params, filepath)

    @staticmethod
    def load(filepath, device_to_load):
        params = torch.load(filepath, map_location=lambda storage, loc: storage)
        model = CRF(params['sent_vocab'], params['tag_vocab'], **params['args'])
        model.load_state_dict(params['state_dict'])
        model.to(device_to_load)
        return model

    @property
    def device(self):
        return self.embedding.weight.device


def main():
    sent_vocab = Vocab.load('../vocab/sent_vocab.json')
    tag_vocab = Vocab.load('../vocab/tag_vocab.json')
    train_data, dev_data = utils.generate_train_dev_dataset('../data/NER-train-utf8.txt', sent_vocab, tag_vocab)
    device = torch.device('cpu')
    model = CRF(sent_vocab, tag_vocab)
    model.to(device)
    model.save('../trained_model/crf_model.pth')
    model = CRF.load('../trained_model/crf_model.pth', device)


if __name__ == '__main__':
    main()