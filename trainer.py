import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import tensorflow as tf
import tensorflow_hub as hub

import config
from data_utils import get_loader, eta, user_friendly_time, progress_bar, time_since
from model import Seq2seq

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

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

        print("load train data")
        self.train_loader = get_loader(config.train_src_file,
                                       config.train_trg_file,
                                       word2idx,
                                       batch_size=config.batch_size,
                                       debug=config.debug)
        self.dev_loader = get_loader(config.dev_src_file,
                                     config.dev_trg_file,
                                     word2idx,
                                     batch_size=config.batch_size,
                                     debug=True,
                                     num=1000)

        train_dir = os.path.join("./save", config.exp_name)
        self.model_dir = os.path.join(train_dir, "train_%d" % int(time.strftime("%m%d%H%M%S")))
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self.model = Seq2seq(embedding, model_path=model_path)
        params =list(self.model.decoder.parameters())

        self.lr = config.lr
        # self.optim = optim.SGD(filter(lambda p: p.requires_grad, params), self.lr, momentum=0.8)
        # self.optim = optim.SGD(filter(lambda p: p.requires_grad, params), self.lr)
        self.optim = optim.Adam(filter(lambda p: p.requires_grad, params), self.lr)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

    def save_model(self, loss, epoch):
        state_dict = {
            "epoch": epoch,
            "current_loss": loss,
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
                        #self.save_model(batch_loss, 1000 + epoch)
                batch_loss = self.step(train_data)

                self.optim.zero_grad()
                batch_loss.backward()
                # gradient clipping
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
        trg_seq, ext_trg_seq, trg_len, oov_list, src_seq = train_data

        sents = [" ".join([self.idx2tok[i] for i in s[1:-1]]) for s in src_seq.tolist()]
        print(sents)
    
        use_vec = embed(sents).numpy()
        use_vec = torch.from_numpy(use_vec).float().to(config.device).unsqueeze(0)
        print(use_vec.size())

        if config.use_gpu:
            trg_seq = trg_seq.to(config.device)
            ext_trg_seq = ext_trg_seq.to(config.device)

        use_doubled = torch.cat([use_vec, use_vec], axis=0)
        encode_states = (use_doubled, use_doubled)
        print(trg_seq.size())

        sos_trg = trg_seq[:, :-1]
        eos_trg = trg_seq[:, 1:]

        logits = self.model.decoder(sos_trg, encode_states)
        batch_size, nsteps, _ = logits.size()
        preds = logits.view(batch_size * nsteps, -1)
        targets = eos_trg.contiguous().view(-1)
        loss = self.criterion(preds, targets)
        return loss

    def evaluate(self, msg):
        self.model.eval_mode()
        val_losses = []
        for i, val_data in tqdm(enumerate(self.dev_loader, start=1)):
            with torch.no_grad():
                val_batch_loss = self.step(val_data)
                val_losses.append(val_batch_loss.item())
        # go back to train mode
        self.model.train_mode()
        val_loss = np.mean(val_losses)

        return val_loss
