import torch
import torch.nn as nn
from vocab import Vocab
import utils
import math

class TransformerCRF(nn.Module):
    def __init__(self, sent_vocab, tag_vocab, dropout_rate=0.5, embed_size=256, hidden_size=256,
                 nhead=8, num_layers=6, dim_feedforward=2048):
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

        # Replace LSTM with Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            batch_first=True  # input shape: (batch, seq_len, features)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.hidden2emit_score = nn.Linear(embed_size, len(self.tag_vocab))
        # transition[k, j] is the score of transitioning from tag k -> tag j
        self.transition = nn.Parameter(torch.randn(len(self.tag_vocab), len(self.tag_vocab)))  # shape: (K, K)

    def forward(self, sentences, tags, sen_lengths):
        """
        Args:
            sentences (tensor): sentences, shape (b, len)
            tags (tensor): corresponding tags, shape (b, len)
            sen_lengths (list): sentence lengths
        Returns:
            loss (tensor): loss on the batch, shape (b,)
        """
        # mask: True for real tokens, False for PAD
        mask = (sentences != self.sent_vocab[self.sent_vocab.PAD]).to(self.device)  # shape: (b, len)

        # Transformer expects src_key_padding_mask with True in positions that should be masked (i.e., PAD positions)
        src_key_padding_mask = ~mask  # True where PAD

        # embedding + positional encoding
        embeddings = self.embedding(sentences)  # shape: (b, len, e)
        # pos_encoding expects (seq_len, batch, d_model)
        embeddings = self.pos_encoding(embeddings.transpose(0, 1)).transpose(0, 1)  # back to (b, len, e)

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
        Viterbi decode per sentence.

        Args:
            sentences (tensor): shape (b, len)
            sen_lengths (list or tensor): lengths for each sentence in the batch
        Returns:
            tags (list[list[int]]): predicted tag ids (one list per sentence)
        """
        device = self.device
        batch_size = sentences.shape[0]
        mask = (sentences != self.sent_vocab[self.sent_vocab.PAD]).to(device)  # (b, len)
        src_key_padding_mask = ~mask

        embeddings = self.embedding(sentences)  # (b, len, e)
        embeddings = self.pos_encoding(embeddings.transpose(0, 1)).transpose(0, 1)
        emit_score = self.encode(embeddings, src_key_padding_mask)  # (b, len, K)

        K = len(self.tag_vocab)
        emit_score = emit_score.to(device)

        # Ensure sen_lengths is a list of ints
        if isinstance(sen_lengths, torch.Tensor):
            sen_lengths = sen_lengths.cpu().tolist()
        sen_lengths = [int(x) for x in sen_lengths]

        results = []
        # Do Viterbi per-sentence (safer and simpler)
        for b in range(batch_size):
            L = sen_lengths[b]
            if L == 0:
                results.append([])
                continue
            # scores for time 0
            scores = emit_score[b, 0].detach().clone()  # (K,)
            backpointers = []  # list of tensors of shape (K,) storing argmax prev tag for each current tag

            for t in range(1, L):
                emit_t = emit_score[b, t]  # (K,)
                # scores.unsqueeze(1): (K,1), transition: (K,K) where transition[k,j]
                # all_scores[k, j] = scores[k] + transition[k, j]
                all_scores = scores.unsqueeze(1) + self.transition  # (K, K)
                # For each current tag j we need max over previous k
                max_scores, argmax_prev = all_scores.max(dim=0)  # both (K,)
                # update scores for current time t
                scores = max_scores + emit_t  # (K,)
                backpointers.append(argmax_prev.detach().cpu())  # store on CPU for easy indexing later

            # backtrace
            last_tag = int(torch.argmax(scores).item())
            path = [last_tag]
            # iterate backpointers in reverse
            for bp in reversed(backpointers):
                last_tag = int(bp[last_tag].item())
                path.append(last_tag)
            path.reverse()  # now path length == L
            results.append(path)

        return results

    # cal_loss 方法保持不变
    def cal_loss(self, tags, mask, emit_score):
        """ Calculate CRF loss
        Args:
            tags (tensor): a batch of tags, shape (b, len)
            mask (tensor): mask for the tags, shape (b, len), values in PAD position is 0 (bool tensor)
            emit_score (tensor): emit matrix, shape (b, len, K)
        Returns:
            loss (tensor): loss of the batch, shape (b,)
        """
        batch_size, sent_len = tags.shape
        # calculate score for the tags
        score = torch.gather(emit_score, dim=2, index=tags.unsqueeze(dim=2)).squeeze(dim=2)  # shape: (b, len)
        score[:, 1:] += self.transition[tags[:, :-1], tags[:, 1:]]
        total_score = (score * mask.type(torch.float)).sum(dim=1)  # shape: (b,)
        # calculate the scaling factor (log-sum-exp over all tag sequences)
        d = torch.unsqueeze(emit_score[:, 0], dim=1)  # shape: (b, 1, K)
        for i in range(1, sent_len):
            n_unfinished = int(mask[:, i].sum().item())
            if n_unfinished == 0:
                break
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
        pe = pe.unsqueeze(0).transpose(0, 1)  # shape: (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x expected shape: (seq_len, batch, d_model)
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
