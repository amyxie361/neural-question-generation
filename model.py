import config
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch_scatter import scatter_max
from data_utils import UNK_ID, PAD_ID

INF = 1e12

class Encoder(nn.Module):
    def __init__(self, embeddings, vocab_size, embedding_size, hidden_size, num_layers, dropout):
        super(Encoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        if config.use_tag:
            self.tag_embedding = nn.Embedding(3, 3)
            lstm_input_size = embedding_size + 3
        else:
            lstm_input_size = embedding_size

        if embeddings is not None:
            self.embedding = nn.Embedding(vocab_size, embedding_size). \
                from_pretrained(embeddings, freeze=config.freeze_embedding)

        self.num_layers = num_layers
        if self.num_layers == 1:
            dropout = 0.0
        self.lstm = nn.LSTM(lstm_input_size, hidden_size, dropout=dropout,
                            num_layers=num_layers, bidirectional=True, batch_first=True)
        self.linear_trans = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.update_layer = nn.Linear(4 * hidden_size, 2 * hidden_size, bias=False)
        self.gate = nn.Linear(4 * hidden_size, 2 * hidden_size, bias=False)

    def gated_self_attn(self, queries, memories, mask):
        # queries: [b,t,d]
        # memories: [b,t,d]
        # mask: [b,t]
        energies = torch.matmul(queries, memories.transpose(1, 2))  # [b, t, t]
        energies = energies.masked_fill(mask.unsqueeze(1), value=-1e12)
        scores = F.softmax(energies, dim=2)
        context = torch.matmul(scores, queries)
        inputs = torch.cat([queries, context], dim=2)
        f_t = torch.tanh(self.update_layer(inputs))
        g_t = torch.sigmoid(self.gate(inputs))
        updated_output = g_t * f_t + (1 - g_t) * queries

        return updated_output

    def forward(self, src_seq, src_len, tag_seq):
        embedded = self.embedding(src_seq)
        if config.use_tag and tag_seq is not None:
            tag_embedded = self.tag_embedding(tag_seq)
            embedded = torch.cat((embedded, tag_embedded), dim=2)
        packed = pack_padded_sequence(embedded, src_len, batch_first=True)
        outputs, states = self.lstm(packed)  # states : tuple of [4, b, d]
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)  # [b, t, d]
        h, c = states

        # self attention
        mask = (src_seq == 0).bool()
        memories = self.linear_trans(outputs)
        outputs = self.gated_self_attn(outputs, memories, mask)

        _, b, d = h.size()
        h = h.view(2, 2, b, d)  # [n_layers, bi, b, d]
        h = torch.cat((h[:, 0, :, :], h[:, 1, :, :]), dim=-1)

        c = c.view(2, 2, b, d)
        c = torch.cat((c[:, 0, :, :], c[:, 1, :, :]), dim=-1)
        concat_states = (h, c)
        return outputs, concat_states

class TreeEncoder(nn.Module):
    def __init__(self, embeddings,vocab_size, embedding_size, in_dim, mem_dim):
        super(TreeEncoder, self).__init__()

        self.emb = nn.Embedding(config.vocab_size, config.hidden_size,
                                padding_idx=PAD_ID, sparse=config.sparsity)

        if embeddings is not None:
             self.embedding = nn.Embedding(vocab_size, embedding_size). \
                     from_pretrained(embeddings, freeze=config.freeze_embedding)

        self.in_dim = in_dim
        self.mem_dim = mem_dim

        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)

    def node_forward(self, inputs, child_c, child_h):
        child_h_sum = torch.sum(child_h, dim=0, keepdim=True)
        # print(inputs, inputs.size())
        iou = self.ioux(inputs) + self.iouh(child_h_sum)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = F.sigmoid(i), F.sigmoid(o), F.tanh(u)

        f = F.sigmoid(
            self.fh(child_h) +
            self.fx(inputs).repeat(len(child_h), 1)
        )
        fc = torch.mul(f, child_c)

        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, F.tanh(c))
        return o, c, h

    def forward(self, tree, inputs):
        embed = self.emb(inputs)[0]
        for idx in range(tree.num_children):
            self.forward(tree.children[idx], inputs)

        if tree.num_children == 0:
            child_c = embed[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
            #child_c = torch.zeros([1, self.mem_dim], dtype=torch.long)
            child_h = embed[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
            #child_h = torch.zeros([1, self.mem_dim], dtype=torch.long)
        else:
            child_o, child_c, child_h = zip(*map(lambda x: x.state, tree.children))
            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)
        tree.state = self.node_forward(embed[tree.idx], child_c, child_h)
        return tree.state


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
        self.encoder_trans = nn.Linear(2 * hidden_size, hidden_size)
        self.reduce_layer = nn.Linear(embedding_size + hidden_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True,
                            num_layers=num_layers, bidirectional=False, dropout=dropout)
        self.concat_layer = nn.Linear(2 * hidden_size, hidden_size)
        self.logit_layer = nn.Linear(hidden_size, vocab_size)

    @staticmethod
    def attention(query, memories, mask):
        # query : [b, 1, d]
        energy = torch.matmul(query, memories.transpose(1, 2))  # [b, 1, t]
        energy = energy.squeeze(1).masked_fill(mask, value=-1e12)
        attn_dist = F.softmax(energy, dim=1).unsqueeze(dim=1)  # [b, 1, t]
        context_vector = torch.matmul(attn_dist, memories)  # [b, 1, d]

        return context_vector, energy

    def get_encoder_features(self, encoder_outputs, tree_output):
        encoder_outputs = torch.cat([tree_output, encoder_outputs[0]], dim=-1).unsqueeze(0)
        return self.encoder_trans(encoder_outputs)

    def forward(self, trg_seq, ext_src_seq, init_states, tree_output, tree_enc_states,  encoder_outputs, encoder_mask):
        # trg_seq : [b,t]
        # init_states : [2,b,d]
        # encoder_outputs : [b,t,d]
        # init_states : a tuple of [2, b, d]

        batch_size, max_len = trg_seq.size()
        hidden_size = encoder_outputs.size(-1)
        memories = self.get_encoder_features(encoder_outputs, tree_output)
        # print(tree_output.size(), encoder_outputs[0].size(), memories.size())
        logits = []
        prev_states = init_states
        prev_context = torch.zeros((batch_size, 1, hidden_size), device=config.device)
        for i in range(max_len):
            y_i = trg_seq[:, i].unsqueeze(1)  # [b, 1]
            embedded = self.embedding(y_i)  # [b, 1, d]
            lstm_inputs = self.reduce_layer(torch.cat([embedded, prev_context], dim=2))
            output, states = self.lstm(lstm_inputs, prev_states)
            # encoder-decoder attention
            context, energy = self.attention(output, memories, encoder_mask)
            concat_input = torch.cat((output, context), dim=2).squeeze(dim=1)
            logit_input = torch.tanh(self.concat_layer(concat_input))
            logit = self.logit_layer(logit_input)  # [b, |V|]

            # maxout pointer network
            if config.use_pointer:
                num_oov = max(torch.max(ext_src_seq - self.vocab_size + 1), 0)
                zeros = torch.zeros((batch_size, num_oov), device=config.device)
                extended_logit = torch.cat([logit, zeros], dim=1)
                out = torch.zeros_like(extended_logit) - INF
                out, _ = scatter_max(energy, ext_src_seq, out=out)
                out = out.masked_fill(out == -INF, 0)
                logit = extended_logit + out
                logit = logit.masked_fill(logit == 0, -INF)

            logits.append(logit)
            # update prev state and context
            prev_states = states
            prev_context = context

        logits = torch.stack(logits, dim=1)  # [b, t, |V|]

        return logits

    def decode(self, y, ext_x, prev_states, prev_context, encoder_features, encoder_mask, tree_enc_states):
        # forward one step lstm
        # y : [b]

        embedded = self.embedding(y.unsqueeze(1))
        lstm_inputs = self.reduce_layer(torch.cat([embedded, prev_context], dim=2))
        # lstm_inputs = self.reduce_layer(torch.cat([embedded, prev_context], dim=2))
        output, states = self.lstm(lstm_inputs, prev_states)
        context, energy = self.attention(output, encoder_features, encoder_mask)
        concat_input = torch.cat((output, context), dim=2).squeeze(dim=1)
        logit_input = torch.tanh(self.concat_layer(concat_input))
        logit = self.logit_layer(logit_input)  # [b, |V|]

        if config.use_pointer:
            batch_size = y.size(0)
            num_oov = max(torch.max(ext_x - self.vocab_size + 1), 0)
            zeros = torch.zeros((batch_size, num_oov), device=config.device)
            extended_logit = torch.cat([logit, zeros], dim=1)
            out = torch.zeros_like(extended_logit) - INF
            out, _ = scatter_max(energy, ext_x, out=out)
            out = out.masked_fill(out == -INF, 0)
            logit = extended_logit + out
            logit = logit.masked_fill(logit == -INF, 0)
            # forcing UNK prob 0
            logit[:, UNK_ID] = -INF

        return logit, states, context


class Seq2seq(nn.Module):
    def __init__(self, embedding=None, is_eval=False, model_path=None):
        super(Seq2seq, self).__init__()
        encoder = Encoder(embedding, config.vocab_size,
                          config.embedding_size, config.hidden_size,
                          config.num_layers,
                          config.dropout)
        decoder = Decoder(embedding, config.vocab_size,
                          config.embedding_size, 2 * config.hidden_size,
                          config.num_layers,
                          config.dropout)

        if config.use_gpu and torch.cuda.is_available():
            device = torch.device(config.device)
            encoder = encoder.to(device)
            decoder = decoder.to(device)

        self.encoder = encoder
        self.decoder = decoder

        if is_eval:
            self.eval_mode()

        if model_path is not None:
            ckpt = torch.load(model_path)
            self.encoder.load_state_dict(ckpt["encoder_state_dict"])
            self.decoder.load_state_dict(ckpt["decoder_state_dict"])

    def eval_mode(self):
        self.encoder = self.encoder.eval()
        self.decoder = self.decoder.eval()

    def train_mode(self):
        self.encoder = self.encoder.train()
        self.decoder = self.decoder.train()


class SeqTree2seq(nn.Module):
    def __init__(self, embedding=None, is_eval=False, model_path=None):
        super(SeqTree2seq, self).__init__()
        utterance_encoder = Encoder(embedding, config.vocab_size,
                          config.embedding_size, config.hidden_size,
                          config.num_layers,
                          config.dropout)

        tree_encoder = TreeEncoder(embedding, config.vocab_size, config.embedding_size, config.hidden_size, config.hidden_size) ## todo: check tree dimension

        decoder = Decoder(embedding, config.vocab_size, #todo: check decoder dimension
                          config.embedding_size, 2 * config.hidden_size,
                          config.num_layers,
                          config.dropout)

        if config.use_gpu and torch.cuda.is_available():
            device = torch.device(config.device)
            utterance_encoder = utterance_encoder.to(device)
            tree_encoder = tree_encoder.to(device)
            decoder = decoder.to(device)

        self.utterance_encoder = utterance_encoder
        self.tree_encoder = tree_encoder
        self.decoder = decoder

        if is_eval:
            self.eval_mode()

        if model_path is not None:
            ckpt = torch.load(model_path)
            self.utterance_encoder.load_state_dict(ckpt["utterance_encoder_state_dict"])
            self.tree_encoder.load_state_dict(ckpt["tree_encoder_state_dict"])
            self.decoder.load_state_dict(ckpt["decoder_state_dict"])

    def eval_mode(self):
        self.utterance_encoder = self.utterance_encoder.eval()
        self.tree_encoder = self.tree_encoder.eval()
        self.decoder = self.decoder.eval()

    def train_mode(self):
        self.utterance_encoder = self.utterance_encoder.train()
        self.tree_encoder = self.tree_encoder.train()
        self.decoder = self.decoder.train()
