import math
import torch
import torch.nn as nn

from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pack_sequence,
    pad_sequence,
    pad_packed_sequence,
)

from convert import get_position
from ucca.layer0 import Terminal

from torch.nn import TransformerEncoder, TransformerEncoderLayer
from decoder import MLP, Biaffine

class LSTM_Encoder(nn.Module):
    def __init__(self, model, vocab, config):
        super(LSTM_Encoder, self).__init__()
        self.bert = model
        self.vocab = vocab
        self.config = config
        
        
        self.word_embedding = nn.Embedding(vocab.num_train_word, config.word_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(vocab.num_pos, config.pos_dim, padding_idx=0)
        self.dep_embedding = nn.Embedding(vocab.num_dep, config.dep_dim, padding_idx=0)
        self.ent_embedding = nn.Embedding(vocab.num_ent, config.ent_dim, padding_idx=0)
        self.ent_iob_embedding = nn.Embedding(vocab.num_ent_iob, config.ent_iob_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(
            input_size=768+config.word_dim + config.pos_dim + config.dep_dim + config.ent_dim + config.ent_iob_dim,
            hidden_size=config.lstm_dim // 2,
            bidirectional=True,
            num_layers=config.lstm_layer,
            dropout=config.lstm_drop
            )
        self.emb_drop = nn.Dropout(config.emb_drop)
        self.lstm_dim = config.lstm_dim
        self.reset_parameters()
    
    def reset_parameters(self):
        self.word_embedding.weight.data.zero_()
        
    def forward(self, word, word_idxs, pos_idxs, dep_idxs, ent_idxs, ent_iob_idxs):
        mask = pos_idxs.ne(self.vocab.PAD_index)
        sen_lens = mask.sum(1)
        sorted_lens, sorted_idx = torch.sort(sen_lens, dim=0, descending=True)
        reverse_idx = torch.sort(sorted_idx, dim=0)[1]
        max_len = sorted_lens[0]

        word = word[:, :max_len]
        word_idxs = word_idxs[:, :max_len]
        pos_idxs = pos_idxs[:, :max_len]
        dep_idxs = dep_idxs[:, :max_len]
        ent_idxs = ent_idxs[:, :max_len]
        ent_iob_idxs = ent_iob_idxs[:, :max_len]
        mask = mask[:, :max_len]

        outputs = self.bert(word)
        
        word_emb_extra = outputs[0]
        word_emb = self.word_embedding(word_idxs.masked_fill_(word_idxs.ge(self.word_embedding.num_embeddings),
                               self.vocab.UNK_index))
        pos_emb = self.pos_embedding(pos_idxs)
        dep_emb = self.dep_embedding(dep_idxs)
        ent_emb = self.ent_embedding(ent_idxs)
        ent_iob_emb = self.ent_iob_embedding(ent_iob_idxs)

        emb = torch.cat((word_emb_extra, word_emb, pos_emb, dep_emb, ent_emb, ent_iob_emb), -1)
        emb = self.emb_drop(emb)

        emb = emb[sorted_idx]
        lstm_input = pack_padded_sequence(emb, sorted_lens, batch_first=True)

        r_out, _ = self.lstm(lstm_input, None)

        lstm_out, _ = pad_packed_sequence(r_out, batch_first=True)

        # get all span vectors
        x = lstm_out[reverse_idx].transpose(0, 1)
        x = x.unsqueeze(1) - x
        x_forward, x_backward = x.chunk(2, dim=-1)

        mask = (mask & word_idxs.ne(self.vocab.STOP_index))[:, :-1]
        mask = mask.unsqueeze(1) & mask.new_ones(max_len - 1, max_len - 1).triu(1)
        lens = mask.sum((1, 2))
        x_forward = x_forward[:-1, :-1].permute(2, 1, 0, 3)
        x_backward = x_backward[1:, 1:].permute(2, 0, 1, 3)
        x_span = torch.cat([x_forward[mask], x_backward[mask]], -1)
        x_span = pad_sequence(torch.split(x_span, lens.tolist()), True)

        return x_span, (sen_lens - 2).tolist()
    
    
        
class Transformer_Encoder(nn.Module):
    def __init__(self, model, vocab, config):
        
        super(Transformer_Encoder, self).__init__()
        self.bert = model
        self.vocab = vocab
        self.config = config
        
        # self.ext_word_embedding
        self.word_embedding = nn.Embedding(vocab.num_train_word, config.word_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(vocab.num_pos, config.pos_dim)
        self.dep_embedding = nn.Embedding(vocab.num_dep, config.dep_dim)
        self.ent_embedding = nn.Embedding(vocab.num_ent, config.ent_dim)
        self.ent_iob_embedding = nn.Embedding(vocab.num_ent_iob, config.ent_iob_dim)        
        
        self.emb_drop = nn.Dropout(config.emb_drop)
        input_size = 768 + config.word_dim + config.pos_dim + config.dep_dim + config.ent_dim + config.ent_iob_dim

        self.pos_encoder = PositionalEncoding(input_size, config.transformer_dropout)
        encoder_layers = TransformerEncoderLayer(input_size, config.heads, config.n_hidden, config.transformer_dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, config.n_layers)
        self.ninp = input_size

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

        
    def forward(self,  word, word_idxs, pos_idxs, dep_idxs, ent_idxs, ent_iob_idxs):
        mask = pos_idxs.ne(self.vocab.PAD_index)
        sen_lens = mask.sum(1)
        sorted_lens, sorted_idx = torch.sort(sen_lens, dim=0, descending=True)
        reverse_idx = torch.sort(sorted_idx, dim=0)[1]
        max_len = sorted_lens[0]

        word = word[:, :max_len]
        word_idxs = word_idxs[:, :max_len]
        pos_idxs = pos_idxs[:, :max_len]
        dep_idxs = dep_idxs[:, :max_len]
        ent_idxs = ent_idxs[:, :max_len]
        ent_iob_idxs = ent_iob_idxs[:, :max_len]
        mask = mask[:, :max_len]

        outputs = self.bert(word)
        
        word_emb_extra = outputs[0]
        word_emb = self.word_embedding(word_idxs.masked_fill_(word_idxs.ge(self.word_embedding.num_embeddings),
                               self.vocab.UNK_index))
        pos_emb = self.pos_embedding(pos_idxs)
        dep_emb = self.dep_embedding(dep_idxs)
        ent_emb = self.ent_embedding(ent_idxs)
        ent_iob_emb = self.ent_iob_embedding(ent_iob_idxs)

        emb = torch.cat((word_emb_extra, word_emb, pos_emb, dep_emb, ent_emb, ent_iob_emb), -1)
        emb = self.emb_drop(emb)

        emb = emb[sorted_idx]
        src = emb *  math.sqrt(self.ninp)

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)

        # get all span vectors
        x = output[reverse_idx].transpose(0, 1)
        x = x.unsqueeze(1) - x
        x_forward, x_backward = x.chunk(2, dim=-1)

        mask = (mask & word_idxs.ne(self.vocab.STOP_index))[:, :-1]
        mask = mask.unsqueeze(1) & mask.new_ones(max_len - 1, max_len - 1).triu(1)
        lens = mask.sum((1, 2))
        x_forward = x_forward[:-1, :-1].permute(2, 1, 0, 3)
        x_backward = x_backward[1:, 1:].permute(2, 0, 1, 3)

        x_span = torch.cat([x_forward[mask], x_backward[mask]], -1)
        x_span = pad_sequence(torch.split(x_span, lens.tolist()), True)

        return x_span, (sen_lens - 2).tolist()
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
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
 
    
class Remote_Parser(nn.Module):
    def __init__(self, vocab, lstm_dim, mlp_label_dim):
        super(Remote_Parser, self).__init__()
        self.vocab = vocab
        self.label_head_mlp = MLP(lstm_dim, mlp_label_dim, nn.ReLU())
        self.label_dep_mlp = MLP(lstm_dim, mlp_label_dim, nn.ReLU())

        self.label_biaffine = Biaffine(
            mlp_label_dim, vocab.num_edge_label, bias_dep=True, bias_head=True
        )

    def forward(self, span_vectors):
        label_head_mlp_out = self.label_head_mlp(span_vectors)
        label_dep_mlp_out = self.label_dep_mlp(span_vectors)

        label_scores = self.label_biaffine(label_head_mlp_out, label_dep_mlp_out)
        return label_scores

    def score(self, span_vectors, sen_len, all_span):
        span_vectors = [span_vectors[get_position(sen_len, i, j).long()] for i, j in all_span]
        span_vectors = torch.stack(span_vectors)
        label_scores = self.forward(span_vectors.unsqueeze(0))
        return label_scores.squeeze(0).permute(1, 2, 0)

    def get_loss(self, spans, sen_lens, all_nodes, all_remote):
        loss_func = torch.nn.CrossEntropyLoss()
        batch_loss = []
        for i, length in enumerate(sen_lens):
            if len(all_remote[i]) == 0:
                batch_loss.append(0)
                continue
            span_num = (1 + length) * length // 2
            label_scores = self.score(spans[i][:span_num], length, all_nodes[i])
            head, dep, label = all_remote[i]
            batch_loss.append(
                loss_func(
                    label_scores[head.view(-1), dep.view(-1)],
                    label.view(-1).to(spans[i].device),
                )
            )
        return batch_loss

    def predict(self, span, sen_len, all_nodes, remote_head):
        label_scores = self.score(span, sen_len, all_nodes)
        labels = label_scores[remote_head].argmax(dim=-1)
        return labels

    def restore_remote(self, passages, spans, sen_lens):
        def get_span_index(node):
            terminals = node.get_terminals()
            return (terminals[0].position - 1, terminals[-1].position)

        for passage, span, length in zip(passages, spans, sen_lens):
            heads = []
            nodes = passage.layer("1").all
            ndict = {node: i for i, node in enumerate(nodes)}
            span_index = [get_span_index(i) for i in nodes]
            for node in nodes:
                for edge in node._incoming:
                    if "-remote" in edge.tag:
                        heads.append(node)
                        if hasattr(edge, "categories"):
                            edge.categories[0]._tag = edge.categories[0]._tag.strip(
                                "-remote"
                            )
                        else:
                            edge._tag = edge._tag.strip("-remote")
            heads = [ndict[node] for node in heads]

            if len(heads) == 0:
                continue
            else:
                label_scores = self.predict(span, length, span_index, heads)

            for head, label_score in zip(heads, label_scores):
                for i, score in enumerate(label_score):
                    label = self.vocab.id2edge_label(score)
                    if label is not self.vocab.NULL and not nodes[i]._tag == "PNCT":
                        passage.layer("1").add_remote(nodes[i], label, nodes[head])
        return passages