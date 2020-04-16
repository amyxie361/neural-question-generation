import json
import pickle
import time
import os
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
import torch.utils.data as data
from tqdm import tqdm
import nltk

# import stanza
#stanza.download('en')
# nlp = stanza.Pipeline()

import config

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "UNKNOWN"
START_TOKEN = "<s>"
END_TOKEN = "EOS"

PAD_ID = 0
UNK_ID = 1
START_ID = 2
END_ID = 3


def collate_fn(data):
    def merge(sequences):
        lengths = [len(sequence) for sequence in sequences]
        padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    data.sort(key=lambda x: len(x[0]), reverse=True)
    trg_seqs, ext_trg_seqs, oov_lst, src_seqs = zip(*data)

    trg_seqs, trg_len = merge(trg_seqs)
    src_seqs, _ = merge(src_seqs)

    ext_trg_seqs, _ = merge(ext_trg_seqs)

    return trg_seqs, ext_trg_seqs, trg_len, oov_lst, src_seqs

class SQuadDatasetWithTag(data.Dataset):
    def __init__(self, src_file,  trg_file, max_length, word2idx, debug=False, num=config.debug_num):
        src_file = open(src_file, "r")
        trg_file = open(trg_file, "r")

        src_data = []
        count = 0
        for line in src_file:
            sent = json.loads(line)
            src_data.append([s.lower() for s in sent])
            count += 1
            if count > config.data_num:
                break
        trg_data = []
        count = 0
        for line in trg_file:
            sent = json.loads(line)
            trg_data.append([s.lower() for s in sent])
            count += 1
            if count > config.data_num:
                break

        self.trgs = trg_data
        # for example in data:
        #     self.trgs.append(example)
        self.num_seqs = len(self.trgs)
        self.srcs = src_data

        self.max_length = max_length
        self.word2idx = word2idx

        if debug:
            num = config.debug_num 
            self.trgs = self.trgs[:num]
            self.srcs = self.srcs[:num]
            self.num_seqs = num

    def __getitem__(self, index):
        trg_seq = self.trgs[index]
        src_seq = self.srcs[index]

        trg_seq, ext_trg_seq, oov_lst = self.context2ids(trg_seq, self.word2idx)
        src_seq, _, _ = self.context2ids(src_seq, self.word2idx)
        return trg_seq, ext_trg_seq, oov_lst, src_seq

    def __len__(self):
        return self.num_seqs

    def context2ids(self, tokens, word2idx):
        ids = list()
        extended_ids = list()
        oov_lst = list()
        ids.append(word2idx[START_TOKEN])
        extended_ids.append(word2idx[START_TOKEN])
        # START and END token is already in tokens lst
        for token in tokens:
            if token in word2idx:
                ids.append(word2idx[token])
                extended_ids.append(word2idx[token])
            else:
                ids.append(word2idx[UNK_TOKEN])
                if token not in oov_lst:
                    oov_lst.append(token)
                extended_ids.append(len(word2idx) + oov_lst.index(token))
            if len(ids) == self.max_length:
                break

        ids.append(word2idx[END_TOKEN])
        extended_ids.append(word2idx[END_TOKEN])

        ids = torch.Tensor(ids)
        extended_ids = torch.Tensor(extended_ids)

        return ids, extended_ids, oov_lst

def get_loader(src_file, trg_file, word2idx, 
               batch_size, use_tag=False, debug=False, num=config.debug_num, shuffle=False):
    # if use_tag:
    dataset = SQuadDatasetWithTag(src_file, trg_file, config.max_seq_len,
                                  word2idx, debug, num)
    dataloader = data.DataLoader(dataset=dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 collate_fn=collate_fn)

    return dataloader


def make_vocab(src_file, trg_file, output_file, max_vocab_size):
    word2idx = dict()
    word2idx[PAD_TOKEN] = 0
    word2idx[UNK_TOKEN] = 1
    word2idx[START_TOKEN] = 2
    word2idx[END_TOKEN] = 3
    counter = dict()
    with open(src_file, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.split()
            for token in tokens:
                if token in counter:
                    counter[token] += 1
                else:
                    counter[token] = 1
    with open(trg_file, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.split()
            for token in tokens:
                if token in counter:
                    counter[token] += 1
                else:
                    counter[token] = 1

    sorted_vocab = sorted(counter.items(), key=lambda kv: kv[1], reverse=True)
    for i, (word, _) in enumerate(sorted_vocab, start=4):
        if i == max_vocab_size:
            break
        word2idx[word] = i
    with open(output_file, "wb") as f:
        pickle.dump(word2idx, f)

    return word2idx


def make_vocab_from_squad(output_file, counter, max_vocab_size):
    sorted_vocab = sorted(counter.items(), key=lambda kv: kv[1], reverse=True)
    word2idx = dict()
    word2idx[PAD_TOKEN] = 0
    word2idx[UNK_TOKEN] = 1
    word2idx[START_TOKEN] = 2
    word2idx[END_TOKEN] = 3

    for idx, (token, freq) in enumerate(sorted_vocab, start=4):
        if len(word2idx) == max_vocab_size:
            break
        word2idx[token] = idx
    with open(output_file, "wb") as f:
        pickle.dump(word2idx, f)

    return word2idx


def make_embedding(embedding_file, output_file, word2idx):
    word2embedding = dict()
    lines = open(embedding_file, "r", encoding="utf-8").readlines()
    for line in tqdm(lines):
        word_vec = line.split(" ")
        word = word_vec[0]
        vec = np.array(word_vec[1:], dtype=np.float32)
        word2embedding[word] = vec
    embedding = np.zeros((len(word2idx), 300), dtype=np.float32)
    num_oov = 0
    for word, idx in word2idx.items():
        if word in word2embedding:
            embedding[idx] = word2embedding[word]
        else:
            embedding[idx] = word2embedding[UNK_TOKEN]
            num_oov += 1
    print("num OOV : {}".format(num_oov))
    with open(output_file, "wb") as f:
        pickle.dump(embedding, f)
    return embedding


def time_since(t):
    """ Function for time. """
    return time.time() - t


def progress_bar(completed, total, step=5):
    """ Function returning a string progress bar. """
    percent = int((completed / total) * 100)
    bar = '[='
    arrow_reached = False
    for t in range(step, 101, step):
        if arrow_reached:
            bar += ' '
        else:
            if percent // t != 0:
                bar += '='
            else:
                bar = bar[:-1]
                bar += '>'
                arrow_reached = True
    if percent == 100:
        bar = bar[:-1]
        bar += '='
    bar += ']'
    return bar


def user_friendly_time(s):
    """ Display a user friendly time from number of second. """
    s = int(s)
    if s < 60:
        return "{}s".format(s)

    m = s // 60
    s = s % 60
    if m < 60:
        return "{}m {}s".format(m, s)

    h = m // 60
    m = m % 60
    if h < 24:
        return "{}h {}m {}s".format(h, m, s)

    d = h // 24
    h = h % 24
    return "{}d {}h {}m {}s".format(d, h, m, s)


def eta(start, completed, total):
    """ Function returning an ETA. """
    # Computation
    took = time_since(start)
    time_per_step = took / completed
    remaining_steps = total - completed
    remaining_time = time_per_step * remaining_steps

    return user_friendly_time(remaining_time)


def outputids2words(id_list, idx2word, article_oovs=None):
    """
    :param id_list: list of indices
    :param idx2word: dictionary mapping idx to word
    :param article_oovs: list of oov words
    :return: list of words
    """
    words = []
    for idx in id_list:
        try:
            word = idx2word[idx]
        except KeyError:
            if article_oovs is not None:
                article_oov_idx = idx - len(idx2word)
                try:
                    word = article_oovs[article_oov_idx]
                except IndexError:
                    print("there's no such a word in extended vocab")
            else:
                word = idx2word[UNK_ID]
        words.append(word)

    return words


def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print("Token {} cannot be found".format(token))
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)

    return spans


def word_tokenize(tokens):
    return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]


def get_truncated_context(context, answer_text, answer_end, parser):
    # get sentences up to the sentence that contains answer span
    doc = parser(context)
    sentences = doc.sentences  # list of Sentence objects
    sents_text = []

    for sentence in sentences:
        sent = []
        for token in sentence.tokens:
            sent.append(token.text)
        sents_text.append(" ".join(sent))
    sentences = sents_text

    stop_idx = -1
    for idx, sentence in enumerate(sentences):
        if answer_text in sentence:
            chars = " ".join(sentences[:idx + 1])
            if len(chars) >= answer_end:
                stop_idx = idx
                break
    if stop_idx == -1:
        print(answer_text)
        print(context)
    truncated_sentences = sentences[:stop_idx + 1]
    truncated_context = " ".join(truncated_sentences).lower()
    return truncated_context


def tokenize(doc, parser):
    words = []
    sentences = parser(doc).sentences
    for sent in sentences:
        toks = sent.tokens
        for token in toks:
            words.append(token.text.lower())
    return words


def process_file(file_name):
    counter = defaultdict(lambda: 0)
    examples = list()
    total = 0
    with open(file_name, "r") as f:
        source = json.load(f)
        articles = source["data"]
    for article in tqdm(articles):
        for para in article["paragraphs"]:
            context = para["context"].replace("''", '" ').replace("``", '" ').lower()

            context_tokens = word_tokenize(context)
            spans = convert_idx(context, context_tokens)

            for qa in para["qas"]:
                total += 1
                ques = qa["question"].replace("''", '" ').replace("``", '" ').lower()
                ques_tokens = word_tokenize(ques)

                for token in ques_tokens:
                    counter[token] += 1

                y1s, y2s = [], []
                answer_texts = []

                for answer in qa["answers"]:
                    answer_text = answer["text"]
                    answer_start = answer["answer_start"]
                    answer_end = answer_start + len(answer_text)
                    answer_texts.append(answer_text)
                    answer_span = []

                    for token in context_tokens:
                        counter[token] += len(para["qas"])

                    for idx, span in enumerate(spans):
                        if not (answer_end <= span[0] or answer_start >= span[1]):
                            answer_span.append(idx)
                    y1, y2 = answer_span[0], answer_span[-1]
                    y1s.append(y1)
                    y2s.append(y2)

                example = {"context_tokens": context_tokens, "ques_tokens": ques_tokens,
                           "y1s": y1s, "y2s": y2s, "answers": answer_texts}
                examples.append(example)

    return examples, counter


def make_conll_format(examples, src_file, trg_file):
    src_fw = open(src_file, "w")
    trg_fw = open(trg_file, "w")
    for example in tqdm(examples):
        c_tokens = example["context_tokens"]
        if "\n" in c_tokens:
            print(c_tokens)
            print("new line")
        copied_tokens = deepcopy(c_tokens)
        q_tokens = example["ques_tokens"]
        # always select the first candidate answer
        start = example["y1s"][0]
        end = example["y2s"][0]

        for idx in range(start, end + 1):
            token = copied_tokens[idx]
            if idx == start:
                tag = "B_ans"
                copied_tokens[idx] = token + "\t" + tag
            else:
                tag = "I_ans"
                copied_tokens[idx] = token + "\t" + tag

        for token in copied_tokens:
            if "\t" in token:
                src_fw.write(token + "\n")
            else:
                src_fw.write(token + "\t" + "O" + "\n")

        src_fw.write("\n")
        question = " ".join(q_tokens)
        trg_fw.write(question + "\n")

    src_fw.close()
    trg_fw.close()


def make_tags(examples):
    tag_list = []
    tok_list = []
    question_list = []
    tree_list = []
    for example in tqdm(examples):
        q_tokens = example["ques_tokens"]
        c_tokens = example["context_tokens"]
        start = example["y1s"][0]
        end = example["y2s"][0]
        tags = []

        for idx in range(len(c_tokens)):
            if idx == start:
                tag = "B_ans"
            elif (idx < end + 1) and (idx > start):
                tag = "I_ans"
            else:
                tag = "O"
            tags.append(tag)

        tag_list.append(tags)
        tok_list.append(c_tokens)
        question = " ".join(q_tokens)
        question_list.append(question)

        doc = nlp(question)
        dep = doc.sentences[0].dependencies
        toks = [d[2].text for d in dep]
        parents = [d[0].id for d in dep]
        rels = [d[1] for d in dep]
        dep_info = {"sentence": question,
                  "toks": toks,
                  "parents": parents,
                  "rels": rels, }
        tree_list.append(dep_info)

    return tag_list, tok_list, question_list, tree_list


def split_dev(input_file, dev_file, test_file):
    with open(input_file) as f:
        input_file = json.load(f)

    input_data = input_file["data"]

    # split the original SQuAD dev set into new dev / test set
    num_total = len(input_data)
    num_dev = int(num_total * 0.5)

    dev_data = input_data[:num_dev]
    test_data = input_data[num_dev:]

    dev_dict = {"data": dev_data}
    test_dict = {"data": test_data}

    with open(dev_file, "w") as f:
        json.dump(dev_dict, f)

    with open(test_file, "w") as f:
        json.dump(test_dict, f)
