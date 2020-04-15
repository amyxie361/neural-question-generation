import config
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch_scatter import scatter_max
from data_utils import UNK_ID, PAD_ID

INF = 1e12


class Decoder(nn.Module):
    def __init__(self, embeddings, vocab_size, embedding_size, hidden_size, num_layers, dropout):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        if embeddings is not None:
            self.embedding = nn.Embedding(vocab_size, embedding_size). \
                from_pretrained(embeddings, freeze=config.freeze_embedding)

        if num_layers == 1:
            dropout = 0.0
        self.init_trans_h = nn.Linear(config.use_size, hidden_size)
        self.init_trans_c = nn.Linear(config.use_size, hidden_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True,
                            num_layers=num_layers, bidirectional=False, dropout=dropout)
        self.logit_layer = nn.Linear(hidden_size, vocab_size)


    def form_init(self, init_states):
        init_h, init_c = init_states 
        return self.init_trans_h(init_h), self.init_trans_c(init_c)

    def forward(self, trg_seq, init_states):

        batch_size, max_len = trg_seq.size()
        hidden_size = config.hidden_size * 2
        logits = []
        prev_states = self.form_init(init_states)
        for i in range(max_len):
            y_i = trg_seq[:, i].unsqueeze(1)  # [b, 1]
            embedded = self.embedding(y_i)  # [b, 1, d]
            lstm_inputs = embedded
            output, states = self.lstm(lstm_inputs, prev_states)
            # encoder-decoder attention
            logit_input = torch.tanh(output.squeeze(dim=1))
            logit = self.logit_layer(logit_input)  # [b, |V|]

            logits.append(logit)
            # update prev state and context
            prev_states = states
            # prev_context = context

        logits = torch.stack(logits, dim=1)  # [b, t, |V|]

        return logits

    def decode(self, y, prev_states):
        # forward one step lstm
        # y : [b]

        embedded = self.embedding(y.unsqueeze(1))
        lstm_inputs = embedded
        output, states = self.lstm(lstm_inputs, prev_states)
        logit = self.logit_layer(torch.tanh(output.squeeze(dim=1)))  # [b, |V|]

        return logit, states


class Seq2seq(nn.Module):
    def __init__(self, embedding=None, is_eval=False, model_path=None):
        super(Seq2seq, self).__init__()

        decoder = Decoder(embedding, config.vocab_size, #todo: check decoder dimension
                          config.embedding_size, 2 * config.hidden_size,
                          config.num_layers,
                          config.dropout)

        if config.use_gpu and torch.cuda.is_available():
            device = torch.device(config.device)
            decoder = decoder.to(device)

        self.decoder = decoder

        if is_eval:
            self.eval_mode()

        if model_path is not None:
            ckpt = torch.load(model_path)
            self.decoder.load_state_dict(ckpt["decoder_state_dict"])

    def eval_mode(self):
        self.decoder = self.decoder.eval()

    def train_mode(self):
        self.decoder = self.decoder.train()
