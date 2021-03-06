import config
import pickle
from data_utils import make_embedding, make_vocab_from_squad, \
    process_file, make_conll_format, \
    make_vocab, make_tags


def make_sent_dataset():
    squad_path = "/data/yqxie/00_data/squad_v1.1/"
    glove_path = "/data/yqxie/00_data/GloVe/"

    train_src_file = "para-train.txt"
    train_trg_file = "tgt-train.txt"

    embedding_file = "glove.840B.300d.txt"
    embedding = "embedding.pkl"
    word2idx_file = "word2idx.pkl"
    # make vocab file
    word2idx = make_vocab(squad_path + train_src_file, squad_path + train_trg_file, glove_path + word2idx_file, config.vocab_size)
    make_embedding(glove_path + embedding_file, glove_path + embedding, word2idx)


def make_para_dataset():
    glove_path = "/data/yqxie/00_data/GloVe/"
    embedding_file = "glove.840B.300d.txt"
    embedding = "embedding.pkl"
    src_word2idx_file = "word2idx.pkl"

    squad_path = "/data/yqxie/00_data/squad_v1.1/"
    train_squad = "train-v1.1.json"
    dev_squad = "dev-v1.1.json"

    train_src_file = "para-train.txt"
    train_trg_file = "tgt-train.txt"
    dev_src_file = "para-dev.txt"
    dev_trg_file = "tgt-dev.txt"

    test_src_file = "para-test.txt"
    test_trg_file = "tgt-test.txt"

    # pre-process training data
    train_examples, counter = process_file(squad_path + train_squad)
    #train_srcs = make_tags(train_examples)
    make_conll_format(train_examples, train_src_file, train_trg_file)
    word2idx = make_vocab_from_squad(glove_path + src_word2idx_file, counter, config.vocab_size)
    make_embedding(glove_path + embedding_file, glove_path + embedding, word2idx)
    #pickle.dump(train_srcs, open(squad_path + "train_srcs.pkl", 'wb'))

    # split dev into dev and test
    # dev_test_examples, _ = process_file(squad_path + dev_squad)
    # random.shuffle(dev_test_examples)
    # num_dev = len(dev_test_examples) // 2
    # dev_examples = dev_test_examples
    #test_examples = dev_test_examples[num_dev:]
    # make_conll_format(dev_examples, dev_src_file, dev_trg_file)
    #make_conll_format(test_examples, test_src_file, test_trg_file)
    # dev_srcs = make_tags(dev_examples)
    #test_srcs = make_tags(test_examples)
    # pickle.dump(dev_srcs, open(squad_path + "dev_srcs.pkl", "wb"))
    #`pickle.dump(test_srcs, open(squad_path + "test_tags.pkl", "wb"))


if __name__ == "__main__":
    # make_sent_dataset()
    make_para_dataset()
