import torch
from encoder import LSTM_Encoder, Transformer_Encoder, Remote_Parser
from decoder import Chart_Span_Parser, Topdown_Span_Parser, Global_Chart_Span_Parser
from ucca.layer0 import Terminal

from convert import to_UCCA
import utils_ucca

class UCCA_Parser(torch.nn.Module):
    def __init__(self, device, bert_model, vocab, config):
        super(UCCA_Parser, self).__init__()
        self.device = device
        self.vocab = vocab
        self.config = config
        
        if self.config.encoder == "lstm":
            self.shared_encoder = LSTM_Encoder(bert_model, vocab, config)
        elif self.config.encoder == "transformer":
            self.shared_encoder = Transformer_Encoder(bert_model, vocab, config)
            
        if config.type == "chart":
            self.span_parser = Chart_Span_Parser(
                vocab=vocab,
                lstm_dim=config.lstm_dim,
                label_hidden_dim=config.label_hidden,
                drop=config.ffn_drop,
                norm=False if config.encoder=='lstm' else True,
            )
        elif config.type == "top-down":
            self.span_parser = Topdown_Span_Parser(
                vocab=vocab,
                lstm_dim=config.lstm_dim,
                label_hidden_dim=config.label_hidden,
                split_hidden_dim=config.split_hidden,
                drop=config.ffn_drop,
                norm=False,
            )
        elif config.type == "global-chart":
            self.span_parser = Global_Chart_Span_Parser(
                vocab=vocab,
                lstm_dim=config.lstm_dim,
                label_hidden_dim=config.label_hidden,
                drop=config.ffn_drop,
                norm=False,
            )
        self.remote_parser = Remote_Parser(
            vocab=vocab,
            lstm_dim=config.lstm_dim,
            mlp_label_dim=config.mlp_label_dim,
        )

    def parse(self, word, word_idxs, pos_idxs, dep_idxs, ent_idxs, ent_iob_idxs,  passages, trees=None, all_nodes=None, all_remote=None):
        spans, sen_lens = self.shared_encoder(word, word_idxs, pos_idxs, dep_idxs, ent_idxs, ent_iob_idxs)

        if self.training:
            span_loss = self.span_parser.get_loss(spans, sen_lens, trees)
            remote_loss = self.remote_parser.get_loss(spans, sen_lens, all_nodes, all_remote)
            return span_loss, remote_loss
        else:
            predict_trees = self.span_parser.predict(spans, sen_lens)
            predict_passages = [to_UCCA(passage, pred_tree) for passage, pred_tree in zip(passages, predict_trees)]
            predict_passages = self.remote_parser.restore_remote(predict_passages, spans, sen_lens)
            return predict_passages

    @classmethod
    def load(cls, device, vocab_path, config_path, state_path):
            
        state = torch.load(state_path, map_location=device)
        vocab = torch.load(vocab_path)
        config = utils_ucca.get_config(config_path)

        network = cls(vocab, config.ucca)
        network.load_state_dict(state['state_dict'])
        network.to(device)

        return network

    def save(self, fname):
        state = {
            'state_dict': self.state_dict(),
        }
        torch.save(state, fname)