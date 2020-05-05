import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import config
from data_utils import get_loader, eta, user_friendly_time, progress_bar, time_since
from model import SeqTree2seq


class Trainer(object):
    def __init__(self, model_path=None):
        # load dictionary and embedding file
        with open(config.embedding, "rb") as f:
            embedding = pickle.load(f)
            embedding = torch.tensor(embedding,
                                     dtype=torch.float).to(config.device)
        with open(config.word2idx_file, "rb") as f:
            word2idx = pickle.load(f)

        self.tok2idx = word2idx
        self.idx2tok = {idx: tok for tok, idx in self.tok2idx.items()}

        # train, dev loader
        print("load train data")
        self.train_loader = get_loader(config.train_src_file,
                                       # config.train_tag_file,
                                       config.train_tree_file,
                                       word2idx,
                                       config.vocab_file,
                                       use_tag=config.use_tag,
                                       batch_size=1,
                                       debug=config.debug)
        self.dev_loader = get_loader(config.dev_src_file,
                                     # config.dev_tag_file,
                                     config.dev_tree_file,
                                     word2idx,
                                     config.vocab_file,
                                     use_tag=config.use_tag,
                                     batch_size=1,
                                     debug=config.debug,
                                     num=100)

        train_dir = os.path.join("./save", config.exp_name)
        self.model_dir = os.path.join(train_dir, "train_%d" % int(time.strftime("%m%d%H%M%S")))
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self.model = SeqTree2seq(embedding, model_path=model_path)
        params = list(self.model.utterance_encoder.parameters()) \
                 + list(self.model.tree_encoder.parameters()) \
                 + list(self.model.decoder.parameters())

        self.lr = config.lr
        # self.optim = optim.SGD(filter(lambda p: p.requires_grad, params), self.lr, momentum=0.8)
        self.optim = optim.SGD(filter(lambda p: p.requires_grad, params), self.lr)
        # self.optim = optim.Adam(params)
        self.criterion_decoder = nn.CrossEntropyLoss(ignore_index=0)
        self.criterion_encoder = nn.CosineEmbeddingLoss()

    def save_model(self, loss, epoch):
        state_dict = {
            "epoch": epoch,
            "current_loss": loss,
            "utterance_encoder_state_dict": self.model.utterance_encoder.state_dict(),
            "tree_encoder_state_dict": self.model.tree_encoder.state_dict(),
            "decoder_state_dict": self.model.decoder.state_dict()
        }
        loss = round(loss, 2)
        model_save_path = os.path.join(self.model_dir, str(epoch) + "_" + str(loss))
        torch.save(state_dict, model_save_path)

    def train(self):
        batch_num = len(self.train_loader)
        self.model.train_mode()
        best_loss = 1e10
        best_train_loss = 1e10
        batch_loss = 1e9
        for epoch in range(1, config.num_epochs + 1):
            print("epoch {}/{} :".format(epoch, config.num_epochs), end="\r")
            start = time.time()
            # halving the learning rate after epoch 8
            #if epoch >= config.decay_start and epoch % config.decay_step == 0:
            for batch_idx, train_data in enumerate(self.train_loader, start=1):
                if batch_idx % 2000 == 0:
                    if batch_loss >= best_train_loss * 1.05:
                        self.lr *= config.decay_weight
                        state_dict = self.optim.state_dict()
                        for param_group in state_dict["param_groups"]:
                            param_group["lr"] = self.lr
                        self.optim.load_state_dict(state_dict)
                    else:
                        best_train_loss = batch_loss 

            #for batch_idx, train_data in enumerate(self.train_loader, start=1):
                batch_loss = self.step(train_data)

                self.optim.zero_grad()
                batch_loss.backward()
                # gradient clipping
                nn.utils.clip_grad_norm_(self.model.utterance_encoder.parameters(), config.max_grad_norm)
                nn.utils.clip_grad_norm_(self.model.tree_encoder.parameters(), config.max_grad_norm)
                nn.utils.clip_grad_norm_(self.model.decoder.parameters(), config.max_grad_norm)
                self.optim.step()
                batch_loss = batch_loss.detach().item()
                msg = "{}/{} {} - ETA : {} - loss : {:.4f}" \
                    .format(batch_idx, batch_num, progress_bar(batch_idx, batch_num),
                            eta(start, batch_idx, batch_num), batch_loss)
                print(msg, end="\r")
                if batch_idx % 1000 == 0:
                    print(msg)

            val_loss = self.evaluate(msg)
            if val_loss <= best_loss:
                best_loss = val_loss
            self.save_model(val_loss, epoch)

            print("Epoch {} took {} - final loss : {:.4f} - val loss :{:.4f}"
                  .format(epoch, user_friendly_time(time_since(start)), batch_loss, val_loss))

    def step(self, train_data):
        #if config.use_tag:
        src_seq, ext_src_seq, src_len, trg_seq, ext_trg_seq, trg_len, tag_seq, _, tree, sent = train_data
        # else:
        #     src_seq, ext_src_seq, src_len, trg_seq, ext_trg_seq, trg_len, tree, _ = train_data
        #     tag_seq = None
        src_len = torch.tensor(src_len, dtype=torch.long)
        trg_seq = sent
        enc_mask = (src_seq == 0).bool()
        with torch.no_grad():
            y = torch.Tensor([1] * config.batch_size)

        if config.use_gpu:
            src_seq = src_seq.to(config.device)
            ext_src_seq = ext_src_seq.to(config.device)
            src_len = src_len.to(config.device)
            trg_seq = trg_seq.to(config.device)
            ext_trg_seq = ext_trg_seq.to(config.device)
            enc_mask = enc_mask.to(config.device)
            sent = sent.to(config.device)
            y = y.to(config.device)
            if config.use_tag:
                tag_seq = tag_seq.to(config.device)
            else:
                tag_seq = None

        tree = tree[0] # TODO: tree can't be batched
        enc_outputs, enc_states = self.model.utterance_encoder(src_seq, src_len, tag_seq)
        tree_enc_o, tree_enc_c, tree_enc_h = self.model.tree_encoder(tree, sent) #todo construct tree input
        # tree_enc_c_double = torch.cat((tree_enc_c, tree_enc_c), axis=0).unsqueeze(1)
        # tree_enc_h_double = torch.cat((tree_enc_h, tree_enc_h), axis=0).unsqueeze(1)
        enc_h, enc_c = enc_states
        # enc_h = torch.cat([tree_enc_h_double, enc_h], axis=-1)
        # enc_c = torch.cat([tree_enc_c_double, enc_c], axis=-1)
        # encode_states = (enc_h, enc_c)
        encode_states = enc_states

        # todo
        cat_enc_c = torch.cat([enc_c[0], enc_c[1]], axis=-1)
        cat_enc_h = torch.cat([enc_h[0], enc_h[1]], axis=-1)
        print(tree_enc_c.size(), cat_enc_c.size())

        encoder_loss = self.criterion_encoder(tree_enc_c, cat_enc_c, y) + \
                       self.criterion_encoder(tree_enc_h, cat_enc_h, y)

        sos_trg = trg_seq[:, :-1]
        eos_trg = trg_seq[:, 1:]

        if config.use_pointer:
            eos_trg = ext_trg_seq[:, 1:]
        logits = self.model.decoder(sos_trg, ext_src_seq, encode_states, enc_outputs, enc_mask)
        batch_size, nsteps, _ = logits.size()
        preds = logits.view(batch_size * nsteps, -1)
        targets = eos_trg.contiguous().view(-1)

        decoder_loss = self.criterion_decoder(preds, targets)
        total_loss = config.tree_loss_alpha * encoder_loss + decoder_loss
        return total_loss

    def evaluate(self, msg):
        self.model.eval_mode()
        # num_val_batches = len(self.dev_loader)
        val_losses = []
        for i, val_data in tqdm(enumerate(self.dev_loader, start=1)):
            with torch.no_grad():
                val_batch_loss = self.step(val_data)
                val_losses.append(val_batch_loss.item())
                # msg2 = "{} => Evaluating :{}/{}".format(msg, i, num_val_batches)
                # print(msg2, end="\r")
        # go back to train mode
        self.model.train_mode()
        val_loss = np.mean(val_losses)

        return val_loss
