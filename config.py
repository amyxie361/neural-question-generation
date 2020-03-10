# train file
squad_path = "/data/yqxie/00_data/squad_v1.1"
glove_path = "/data/yqxie/00_data/GloVe"
train_src_file = squad_path + "/para-train.txt"
train_trg_file = squad_path + "/tgt-train.txt"
# dev file
dev_src_file = squad_path + "/para-dev.txt"
dev_trg_file = squad_path + "/tgt-dev.txt"
# test file
test_src_file = squad_path + "/para-test.txt"
test_trg_file = squad_path + "/tgt-test.txt"
# embedding and dictionary file
embedding = glove_path + "/embedding.pkl"
word2idx_file = glove_path + "/word2idx.pkl"

# tree file
train_tree_file = "../data/squad_train"
vocab_file = "../data/vocab.txt"
dev_tree_file = "../data/squad_dev"

model_path = "./save/tree/train_model/"
train = True
test = False
device = "cuda:1"
use_gpu = True
debug = True
vocab_size = 45000
freeze_embedding = True

num_epochs = 20
max_seq_len = 400
num_layers = 2
hidden_size = 300
embedding_size = 300
lr = 0.1
batch_size = 64
dropout = 0.3
max_grad_norm = 5.0

use_tag = True
use_pointer = True
beam_size = 10
min_decode_step = 8
max_decode_step = 30
output_dir = "./result/pointer_maxout_ans"
