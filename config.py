# train file
squad_path = "/data/yqxie/00_data/squad_v1.1"
glove_path = "/data/yqxie/00_data/GloVe"
# train_src_file = squad_path + "/train_srcs.pkl"
train_src_file = "/data/yqxie/01_exps/100_paraphrase/paraphrase/train_sent_token.txt"
# train_trg_file = squad_path + "/tgt-train.txt"
# dev file
dev_src_file = train_src_file
# dev_trg_file = squad_path + "/tgt-dev.txt"
# test file
# test_src_file = squad_path + "/para-test.txt"
# test_trg_file = squad_path + "/tgt-test.txt"
# embedding and dictionary file
# embedding = glove_path + "/embedding.pkl"
# word2idx_file = glove_path + "/word2idx.pkl"
embedding = glove_path + "/embedding_en_wiki.pkl"
word2idx_file = glove_path + "/word2idx_en_wiki.pkl"

# tree file
train_tree_file = "../data/squad_train_dependency_parse.txt"
vocab_file = "../data/vocab.txt"
dev_tree_file = "../data/squad_dev_dependency_parse.txt"

exp_name = "paraphrase_new_batch32_0.1m"
model_path = "./save/" + exp_name + "/train_414155505/157_0.67"
train = True
test = False
device = "cuda:2"
use_gpu = True
debug = False
debug_num = 100
data_num = 1e5
vocab_size = 45000
freeze_embedding = False
sparsity = False
use_size = 512

num_epochs = 1000000
max_seq_len = 100
num_layers = 2
hidden_size = 300
embedding_size = 300
lr = 1e-3
decay_start = 8
decay_step = 2
decay_weight = 0.8
batch_size = 32
dropout = 0.3
max_grad_norm = 20.0

use_tag = False
use_pointer = False
beam_size = 8
min_decode_step = 8
max_decode_step = 100
output_dir = "./result/" + exp_name
