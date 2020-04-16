from model import Seq2seq
import os
from data_utils import START_TOKEN, END_ID, get_loader, UNK_ID, outputids2words
import torch
import torch.nn.functional as F
import config
import pickle

import tensorflow as tf
import tensorflow_hub as hub
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

class Hypothesis(object):
    def __init__(self, tokens, log_probs, state):
        self.tokens = tokens
        self.log_probs = log_probs
        self.state = state

    def extend(self, token, log_prob, state):
        h = Hypothesis(tokens=self.tokens + [token],
                       log_probs=self.log_probs + [log_prob],
                       state=state)
        return h

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def avg_log_prob(self):
        return sum(self.log_probs) / len(self.tokens)


class BeamSearcher(object):
    def __init__(self, model_path, output_dir):
        with open(config.word2idx_file, "rb") as f:
            print(config.word2idx_file)
            word2idx = pickle.load(f)

        self.output_dir = output_dir
        self.data_loader = get_loader(config.dev_src_file,
                                      config.dev_trg_file,
                                      word2idx,
                                      batch_size=1,
                                      debug=config.debug,
                                      num=100)

        self.tok2idx = word2idx
        self.idx2tok = {idx: tok for tok, idx in self.tok2idx.items()}
        self.model = Seq2seq(model_path=model_path)
        self.pred_dir = output_dir + "/generated.txt"
        self.golden_dir = output_dir + "/golden.txt"
        self.trg_dir = output_dir + "/origin.txt"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    @staticmethod
    def sort_hypotheses(hypotheses):
        return sorted(hypotheses, key=lambda h: h.avg_log_prob, reverse=True)

    def decode(self):
        pred_fw = open(self.pred_dir, "w")
        golden_fw = open(self.golden_dir, "w")
        trg_fw = open(self.trg_dir, "w")
        for i, eval_data in enumerate(self.data_loader):
            trg_seq, ext_trg_seq, trg_len, oov_lst, src_seq = eval_data
            trg_seq = trg_seq.tolist()[0]
            golden_question = " ".join([self.idx2tok[id_] for id_ in trg_seq[1:-1]])
            print("==========")
            print("golden: ", golden_question)
            print("src: ", " ".join([self.idx2tok[id_] for id_ in src_seq.tolist()[0]]))
            src_sent = [" ".join([self.idx2tok[i] for i in src_seq.tolist()[0][1:-1]])]
            trg_fw.write(src_sent[0] + "\n")
            best_questions = self.beam_search(src_sent)
            best_question = best_questions[0]
            # discard START  token
            #outs = [[int(idx) for idx in q.tokens[1:-1]] for q in best_questions]
            #decodes = [outputids2words(out, self.idx2tok, oov_lst[0]) for out in outs]
            #print(decodes)
            output_indices = [int(idx) for idx in best_question.tokens[1:-1]]
            decoded_words = outputids2words(output_indices, self.idx2tok, oov_lst[0])
            try:
                fst_stop_idx = decoded_words.index(END_ID)
                decoded_words = decoded_words[:fst_stop_idx]
            except ValueError:
                decoded_words = decoded_words
            decoded_words = " ".join(decoded_words)
            print(decoded_words)
            pred_fw.write(decoded_words + "\n")
            golden_fw.write(golden_question + "\n")

        pred_fw.close()
        golden_fw.close()

    def interactive(self, sentence):
        best_questions = self.beam_search([sentence])
        outs = [[int(idx) for idx in q.tokens[1:-1]] for q in best_questions]
        decodes = [outputids2words(out, self.idx2tok, []) for out in outs]
        return decodes


    def beam_search(self, sents):
        prev_context = torch.zeros(1, 1, 2 * config.hidden_size)
        #sents = [" ".join([self.idx2tok[i] for i in sent.tolist()[0][1:-1]])]
        use_vec = embed(sents).numpy()
        print(sents)
        
        if config.use_gpu:
            prev_context = prev_context.to(config.device)
            use_vec = torch.from_numpy(use_vec).float().to(config.device)

        use_doubled = torch.cat([use_vec, use_vec], axis=0).unsqueeze(1)
        h = self.model.decoder.init_trans_h(use_doubled)
        c = self.model.decoder.init_trans_c(use_doubled)
        hypotheses = [Hypothesis(tokens=[self.tok2idx[START_TOKEN]],
                                 log_probs=[0.0],
                                 state=(h[:, 0, :], c[:, 0, :]))]
        # tile enc_outputs, enc_mask for beam search
        num_steps = 0
        results = []
        while num_steps < config.max_decode_step and len(results) < config.beam_size:
            latest_tokens = [h.latest_token for h in hypotheses]
            latest_tokens = [idx if idx < len(self.tok2idx) else UNK_ID for idx in latest_tokens]
            prev_y = torch.LongTensor(latest_tokens).view(-1)

            if config.use_gpu:
                prev_y = prev_y.to(config.device)

            # make batch of which size is beam size
            all_state_h = []
            all_state_c = []
            for h in hypotheses:
                state_h, state_c = h.state  # [num_layers, d]
                all_state_h.append(state_h)
                all_state_c.append(state_c)

            prev_h = torch.stack(all_state_h, dim=1)  # [num_layers, beam, d]
            prev_c = torch.stack(all_state_c, dim=1)  # [num_layers, beam, d]
            prev_states = (prev_h, prev_c)
            # [beam_size, |V|]
            logits, states = self.model.decoder.decode(prev_y, prev_states)
            h_state, c_state = states
            log_probs = F.log_softmax(logits, dim=1)
            top_k_log_probs, top_k_ids \
                = torch.topk(log_probs, config.beam_size * 2, dim=-1)

            all_hypotheses = []
            num_orig_hypotheses = 1 if num_steps == 0 else len(hypotheses)
            for i in range(num_orig_hypotheses):
                h = hypotheses[i]
                state_i = (h_state[:, i, :], c_state[:, i, :])
                for j in range(config.beam_size * 2):
                    new_h = h.extend(token=top_k_ids[i][j].item(),
                                     log_prob=top_k_log_probs[i][j].item(),
                                     state=state_i)
                    all_hypotheses.append(new_h)

            hypotheses = []
            for h in self.sort_hypotheses(all_hypotheses):
                if h.latest_token == END_ID:
                    if num_steps >= config.min_decode_step:
                        results.append(h)
                else:
                    hypotheses.append(h)

                if len(hypotheses) == config.beam_size or len(results) == config.beam_size:
                    break
            num_steps += 1
        if len(results) == 0:
            results = hypotheses
        h_sorted = self.sort_hypotheses(results)

        return h_sorted
