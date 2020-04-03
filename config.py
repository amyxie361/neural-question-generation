# train file
squad_path = "/data/yqxie/00_data/squad_v1.1"
glove_path = "/data/yqxie/00_data/GloVe"
train_src_file = squad_path + "/train_srcs.pkl"
# train_trg_file = squad_path + "/tgt-train.txt"
# dev file
dev_src_file = squad_path + "/dev_srcs.pkl"
# dev_trg_file = squad_path + "/tgt-dev.txt"
# test file
# test_src_file = squad_path + "/para-test.txt"
# test_trg_file = squad_path + "/tgt-test.txt"
# embedding and dictionary file
embedding = glove_path + "/embedding.pkl"
word2idx_file = glove_path + "/word2idx.pkl"

# tree file
train_tree_file = "../data/squad_train_dependency_parse.txt"
vocab_file = "../data/vocab.txt"
dev_tree_file = "../data/squad_dev_dependency_parse.txt"

model_path = "./save/"
exp_name = "use_debug"
model_path = "./save/" + exp_name + "/train_328151914/9_2.35"
train = True
test = False
device = "cuda:0"
use_gpu = True
debug = True
debug_num = 100
vocab_size = 45000
freeze_embedding = True
sparsity = False
use_size = 512

num_epochs = 5
max_seq_len = 400
num_layers = 2
hidden_size = 300
embedding_size = 300
lr = 0.1
decay_start = 8
decay_step = 2
decay_weight = 0.9
batch_size = 1
dropout = 0.3
max_grad_norm = 20.0

use_tag = True
use_pointer = True
beam_size = 3
min_decode_step = 5
max_decode_step = 20
output_dir = "./result/" + exp_name + "_train"
